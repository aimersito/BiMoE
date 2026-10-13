--!native
--!strict
-- Benchmark: costo del forward MonoFiLM en Luau nativo + buffer.
-- Mide SOLO matmuls densos (fc1 285xH, fc2 HxH, fc3 Hx95) = ~95% del costo.
-- FiLM/ReLU/normalizar suman ~5%: multiplica el resultado por 1.05.
--
-- USO (2 min, sin instalar nada):
--   1. Roblox Studio -> nuevo Place base.
--   2. ServerScriptService -> inserta un Script (NO ModuleScript).
--   3. Pega TODO este archivo, dale Play, abre Output (View -> Output).
--   4. Copia la tabla que imprime y pasamela: con TUS numeros sale el
--      techo real de params para tu juego (60Hz = 16.67ms/frame).
--
-- REGLA: inferencia < 2ms = seguro a 60Hz. 2-8ms = a 60Hz con cuidado
-- (o 30Hz + interpolar). > 8ms = solo horneado/destilado.

local IN_DIM = 285
local OUT_DIM = 95
local SIZES = { 56, 128, 256, 512, 1024, 1180, 2000 } -- H: 32k,82k,197k,525k,1.5M,2M,5M params
local ITERS = 60 -- repeticiones por tamano (mediana de 3 rondas)

local function makeLayer(rows: number, cols: number)
	local w = buffer.create(rows * cols * 4)
	for i = 0, rows * cols - 1 do
		buffer.writef32(w, i * 4, (math.random() - 0.5) * 0.1)
	end
	local b = buffer.create(rows * 4)
	for i = 0, rows - 1 do
		buffer.writef32(b, i * 4, 0)
	end
	return w, b
end

-- y = W x + b, buffers planos f32. Locals fuera del loop (inline-cache friendly).
local function dense(w: buffer, b: buffer, x: buffer, y: buffer, rows: number, cols: number)
	local r32 = buffer.readf32
	local w32 = buffer.writef32
	for i = 0, rows - 1 do
		local s = r32(b, i * 4)
		local base = i * cols
		for j = 0, cols - 1 do
			s += r32(w, (base + j) * 4) * r32(x, j * 4)
		end
		w32(y, i * 4, if s > 0 then s else 0) -- ReLU fusionada
	end
end

local function bench(H: number): number
	local w1, b1 = makeLayer(H, IN_DIM)
	local w2, b2 = makeLayer(H, H)
	local w3, b3 = makeLayer(OUT_DIM, H)
	local x = buffer.create(IN_DIM * 4)
	for j = 0, IN_DIM - 1 do
		buffer.writef32(x, j * 4, math.random() - 0.5)
	end
	local h1 = buffer.create(H * 4)
	local h2 = buffer.create(H * 4)
	local y = buffer.create(OUT_DIM * 4)
	-- warmup (native compila + caches calientes)
	for _ = 1, 10 do
		dense(w1, b1, x, h1, H, IN_DIM)
		dense(w2, b2, h1, h2, H, H)
		dense(w3, b3, h2, y, OUT_DIM, H)
	end
	local best = math.huge
	for _ = 1, 3 do
		local t0 = os.clock()
		for _ = 1, ITERS do
			dense(w1, b1, x, h1, H, IN_DIM)
			dense(w2, b2, h1, h2, H, H)
			dense(w3, b3, h2, y, OUT_DIM, H)
		end
		local ms = (os.clock() - t0) / ITERS * 1000 * 1.05 -- +5% FiLM/norma
		if ms < best then
			best = ms
		end
	end
	return best
end

print("=== BiMoE Luau bench (forward MonoFiLM, 1 muestra) ===")
print(string.format("%-6s %-10s %-12s %s", "H", "params~", "ms/inf", "%frame60Hz"))
for _, H in SIZES do
	local macs = H * H + 508 * H + 64
	local ms = bench(H)
	print(string.format("H=%-5d %-10d %-12.2f %.1f%%", H, macs, ms, ms / 16.67 * 100))
end
print("=== fin: copia estas 7 lineas ===")
