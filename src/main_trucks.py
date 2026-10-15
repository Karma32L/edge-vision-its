import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response, render_template_string, jsonify
import threading
import time
import math
from datetime import datetime

app = Flask(__name__)

print("Cargando modelo YOLOv8...")
model = YOLO('yolov8n.pt')

# Clases: 7 = truck, 5 = bus, 2 = car
TARGET_CLASSES = [7, 5, 2]

# Estado global
historial_eventos = []
conteo_colores = {"Rojo": 0, "Azul": 0, "Verde": 0, "Amarillo": 0, "Blanco": 0, "Negro": 0, "Otro": 0}
conteo_tipos = {"Camión": 0, "Autobús": 0, "Auto": 0}
total_vehiculos = 0

# Variables de rastreo anti-duplicados
ultimas_detecciones = []

output_frame = None
lock = threading.Lock()

def detectar_color(crop_img):
    if crop_img.size == 0:
        return "Otro"
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    colores = {
        'Rojo': [((0, 50, 50), (10, 255, 255)), ((170, 50, 50), (180, 255, 255))],
        'Azul': [((100, 100, 50), (140, 255, 255))],
        'Verde': [((36, 50, 50), (86, 255, 255))],
        'Amarillo': [((15, 100, 100), (35, 255, 255))],
        'Blanco': [((0, 0, 180), (180, 30, 255))],
        'Negro': [((0, 0, 0), (180, 255, 40))]
    }
    max_pixels = 0
    color_detectado = "Otro"
    for color_nombre, rangos in colores.items():
        mask = np.zeros(hsv.shape[:2], dtype="uint8")
        for (inferior, superior) in rangos:
            inferior = np.array(inferior, dtype="uint8")
            superior = np.array(superior, dtype="uint8")
            mask_rango = cv2.inRange(hsv, inferior, superior)
            mask = cv2.bitwise_or(mask, mask_rango)
        count = cv2.countNonZero(mask)
        if count > max_pixels:
            max_pixels = count
            color_detectado = color_nombre
    return color_detectado

def es_nuevo_vehiculo(centro_x, centro_y):
    global ultimas_detecciones
    ahora = time.time()
    
    # Limpiar registros antiguos (> 4 segundos sin verse)
    ultimas_detecciones = [d for d in ultimas_detecciones if ahora - d['ultimo_visto'] < 4.0]

    for det in ultimas_detecciones:
        cx_previo, cy_previo = det['centro']
        distancia = math.hypot(centro_x - cx_previo, centro_y - cy_previo)
        
        if distancia < 80:
            det['ultimo_visto'] = ahora
            det['centro'] = (centro_x, centro_y)
            return False
    
    ultimas_detecciones.append({'centro': (centro_x, centro_y), 'ultimo_visto': ahora})
    return True

def process_camera():
    global output_frame, lock, total_vehiculos, conteo_colores, conteo_tipos, historial_eventos
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: No se pudo abrir /dev/video0")
        return

    frame_count = 0
    last_boxes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1

        if frame_count % 2 == 0:
            last_boxes = []
            results = model(frame, imgsz=416, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls_id in TARGET_CLASSES and conf > 0.30:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        centro_x = int((x1 + x2) / 2)
                        centro_y = int((y1 + y2) / 2)
                        
                        crop = frame[y1:y2, x1:x2]
                        color = detectar_color(crop)
                        nombre_clase = "Camión" if cls_id == 7 else ("Autobús" if cls_id == 5 else "Auto")
                        
                        last_boxes.append((x1, y1, x2, y2, color, conf, nombre_clase))

                        if es_nuevo_vehiculo(centro_x, centro_y):
                            hora_actual = datetime.now().strftime("%H:%M:%S")
                            nuevo_evento = {
                                "id": len(historial_eventos) + 1,
                                "hora": hora_actual,
                                "tipo": nombre_clase,
                                "color": color,
                                "confianza": f"{int(conf * 100)}%"
                            }
                            historial_eventos.insert(0, nuevo_evento)
                            if len(historial_eventos) > 20:
                                historial_eventos.pop()
                            
                            total_vehiculos += 1
                            conteo_tipos[nombre_clase] += 1
                            if color in conteo_colores:
                                conteo_colores[color] += 1
                            else:
                                conteo_colores["Otro"] += 1

        for (x1, y1, x2, y2, color, conf, nombre_clase) in last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{nombre_clase} {color} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        with lock:
            output_frame = frame.copy()

def generate():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        time.sleep(0.02)

# --- RUTAS DE NAVEGACIÓN Y VISTAS WEB ---

ESTILOS_BASE = """
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }
    .navbar { display: flex; justify-content: center; gap: 20px; background: #1e1e1e; padding: 12px; border-radius: 8px; margin-bottom: 20px; }
    .navbar a { color: #e0e0e0; text-decoration: none; font-weight: bold; padding: 8px 16px; border-radius: 5px; transition: 0.3s; }
    .navbar a:hover, .navbar a.active { background: #00e676; color: #000; }
    h1 { text-align: center; color: #00e676; margin-bottom: 20px; }
    .container { display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }
    .card { background: #1e1e1e; border-radius: 10px; padding: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }
"""

@app.route("/")
def index():
    html_template = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Raspberry Pi 5 - Monitoreo en Vivo</title>
        <style>
            {ESTILOS_BASE}
            .video-card {{ flex: 1 1 500px; max-width: 660px; text-align: center; }}
            .stats-card {{ flex: 1 1 400px; max-width: 500px; }}
            img {{ width: 100%; border-radius: 8px; border: 2px solid #333; }}
            .stat-box {{ display: flex; justify-content: space-between; background: #2a2a2a; padding: 10px 15px; margin-bottom: 10px; border-radius: 5px; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #333; }}
            th {{ background-color: #2a2a2a; color: #00e676; }}
            tr:hover {{ background-color: #252525; }}
            .badge {{ padding: 3px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold; }}
            .badge-camion {{ background-color: #ff9800; color: #000; }}
            .badge-bus {{ background-color: #2196f3; color: #fff; }}
            .badge-auto {{ background-color: #9c27b0; color: #fff; }}
        </style>
    </head>
    <body>
        <div class="navbar">
            <a href="/" class="active">📹 Monitoreo en Vivo</a>
            <a href="/analytics">📊 Analytics & Métricas</a>
        </div>

        <h1>🚦 Transmisión Edge AI - Raspberry Pi 5</h1>
        <div class="container">
            <div class="card video-card">
                <h2>Video Feed con Inferencia</h2>
                <img src="/video_feed" alt="Video Stream">
            </div>
            <div class="card stats-card">
                <h2>Métricas In citio</h2>
                <div class="stat-box">
                    <span>Total Vehículos Capturados:</span>
                    <span id="total-vehiculos" style="color: #00e676;">0</span>
                </div>

                <h3>Historial Reciente</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Hora</th>
                            <th>Tipo</th>
                            <th>Color</th>
                            <th>Confianza</th>
                        </tr>
                    </thead>
                    <tbody id="tabla-historial">
                        <tr><td colspan="4" style="text-align:center;">Esperando detecciones...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <script>
            async function actualizarStats() {{
                try {{
                    const response = await fetch('/api/stats');
                    const data = await response.json();
                    
                    document.getElementById('total-vehiculos').innerText = data.total_vehiculos;
                    
                    const tabla = document.getElementById('tabla-historial');
                    if (data.historial.length > 0) {{
                        tabla.innerHTML = '';
                        data.historial.forEach(ev => {{
                            let badgeClass = ev.tipo === 'Camión' ? 'badge-camion' : (ev.tipo === 'Autobús' ? 'badge-bus' : 'badge-auto');
                            tabla.innerHTML += `
                                <tr>
                                    <td>${{ev.hora}}</td>
                                    <td><span class="badge ${{badgeClass}}">${{ev.tipo}}</span></td>
                                    <td>${{ev.color}}</td>
                                    <td>${{ev.confianza}}</td>
                                </tr>
                            `;
                        }});
                    }}
                }} catch (e) {{
                    console.error("Error:", e);
                }}
            }}

            setInterval(actualizarStats, 1000);
            actualizarStats();
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route("/analytics")
def analytics():
    html_analytics = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Raspberry Pi 5 - Control de Tráfico Analytics</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            {ESTILOS_BASE}
            .kpi-grid {{ display: flex; gap: 20px; justify-content: center; width: 100%; max-width: 1100px; margin-bottom: 20px; }}
            .kpi-card {{ flex: 1; background: #1e1e1e; border-radius: 10px; padding: 15px; text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }}
            .kpi-title {{ font-size: 0.9em; color: #aaa; margin-bottom: 5px; }}
            .kpi-value {{ font-size: 1.8em; font-weight: bold; color: #00e676; }}
            .chart-card {{ flex: 1 1 450px; max-width: 520px; height: 350px; position: relative; }}
        </style>
    </head>
    <body>
        <div class="navbar">
            <a href="/">📹 Monitoreo en Vivo</a>
            <a href="/analytics" class="active">📊 Analytics & Métricas</a>
        </div>

        <h1>📈 Control de Tráfico Urbano - Analytics</h1>

        <div class="container" style="max-width: 1100px; margin: 0 auto;">
            <!-- Panel de KPIs -->
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-title">Flujo Vehicular Total</div>
                    <div class="kpi-value" id="kpi-total">0</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Camiones / Buses</div>
                    <div class="kpi-value" id="kpi-pesados" style="color: #ff9800;">0</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Color Predominante</div>
                    <div class="kpi-value" id="kpi-color-top" style="color: #2196f3;">-</div>
                </div>
            </div>

            <!-- Gráficas de Análisis -->
            <div class="card chart-card">
                <h3 style="text-align: center; color: #00e676;">Distribución por Tipo de Vehículo</h3>
                <canvas id="chartTipos"></canvas>
            </div>

            <div class="card chart-card">
                <h3 style="text-align: center; color: #00e676;">Análisis por Gama de Color</h3>
                <canvas id="chartColores"></canvas>
            </div>
        </div>

        <script>
            let chartTipos, chartColores;

            function inicializarGraficas() {{
                // Gráfica de Dona (Tipos de Vehículo)
                const ctxTipos = document.getElementById('chartTipos').getContext('2d');
                chartTipos = new Chart(ctxTipos, {{
                    type: 'doughnut',
                    data: {{
                        labels: ['Camión', 'Autobús', 'Auto'],
                        datasets: [{{
                            data: [0, 0, 0],
                            backgroundColor: ['#ff9800', '#2196f3', '#9c27b0']
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }}
                    }}
                }});

                // Gráfica de Barras (Colores)
                const ctxColores = document.getElementById('chartColores').getContext('2d');
                chartColores = new Chart(ctxColores, {{
                    type: 'bar',
                    data: {{
                        labels: ['Rojo', 'Azul', 'Verde', 'Amarillo', 'Blanco', 'Negro', 'Otro'],
                        datasets: [{{
                            label: 'Cantidad',
                            data: [0, 0, 0, 0, 0, 0, 0],
                            backgroundColor: ['#f44336', '#2196f3', '#4caf50', '#ffeb3b', '#ffffff', '#616161', '#9e9e9e']
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            x: {{ ticks: {{ color: '#e0e0e0' }} }},
                            y: {{ ticks: {{ color: '#e0e0e0' }}, beginAtZero: true }}
                        }},
                        plugins: {{ legend: {{ display: false }} }}
                    }}
                }});
            }}

            async function actualizarAnalytics() {{
                try {{
                    const response = await fetch('/api/stats');
                    const data = await response.json();

                    // Actualizar KPIs
                    document.getElementById('kpi-total').innerText = data.total_vehiculos;
                    const pesados = (data.conteo_tipos['Camión'] || 0) + (data.conteo_tipos['Autobús'] || 0);
                    document.getElementById('kpi-pesados').innerText = pesados;

                    // Color predominante
                    let colorTop = '-';
                    let maxVal = 0;
                    for (const [col, cant] of Object.entries(data.conteo_colores)) {{
                        if (cant > maxVal) {{
                            maxVal = cant;
                            colorTop = col;
                        }}
                    }}
                    document.getElementById('kpi-color-top').innerText = colorTop;

                    // Actualizar Gráfica de Dona
                    chartTipos.data.datasets[0].data = [
                        data.conteo_tipos['Camión'] || 0,
                        data.conteo_tipos['Autobús'] || 0,
                        data.conteo_tipos['Auto'] || 0
                    ];
                    chartTipos.update();

                    // Actualizar Gráfica de Barras
                    chartColores.data.datasets[0].data = [
                        data.conteo_colores['Rojo'] || 0,
                        data.conteo_colores['Azul'] || 0,
                        data.conteo_colores['Verde'] || 0,
                        data.conteo_colores['Amarillo'] || 0,
                        data.conteo_colores['Blanco'] || 0,
                        data.conteo_colores['Negro'] || 0,
                        data.conteo_colores['Otro'] || 0
                    ];
                    chartColores.update();

                }} catch (e) {{
                    console.error("Error al obtener datos analíticos:", e);
                }}
            }}

            window.onload = function() {{
                inicializarGraficas();
                actualizarAnalytics();
                setInterval(actualizarAnalytics, 1500);
            }};
        </script>
    </body>
    </html>
    """
    return render_template_string(html_analytics)

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/stats")
def api_stats():
    return jsonify({
        "total_vehiculos": total_vehiculos,
        "conteo_colores": conteo_colores,
        "conteo_tipos": conteo_tipos,
        "historial": historial_eventos
    })

if __name__ == "__main__":
    t = threading.Thread(target=process_camera)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
