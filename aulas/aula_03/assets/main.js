const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const gestureEl = document.getElementById('gesture_text');
const gestureImg = document.getElementById('gesture_img');

const rngQuality = document.getElementById('rng_quality');
const valQuality = document.getElementById('val_quality');
const chkLandmarks = document.getElementById('chk_landmarks');
const valFps = document.getElementById('val_fps');

if (rngQuality && valQuality) {
    rngQuality.addEventListener('input', (e) => {
        valQuality.textContent = e.target.value + '%';
    });
}

// Câmera escondida para captura
const video = document.createElement('video');
video.setAttribute('playsinline', '');
video.setAttribute('autoplay', '');
video.style.opacity = '0';
video.style.position = 'absolute';
video.style.pointerEvents = 'none';
video.style.width = '1px';
video.style.height = '1px';
document.body.appendChild(video);

// Canvas escondido recebe as dimensões reais para não perder qualidade da câmera
const off = document.createElement('canvas');
off.width = 640; off.height = 480;
const offCtx = off.getContext('2d');

let ws = null;
let busy = false;
let videoReady = false;

video.addEventListener('playing', () => {
    videoReady = true;
    if(ws && ws.readyState === 1) sendFrame();
});

async function start() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({video: {width:640, height:480}});
        video.srcObject = stream;
        await video.play();
    } catch(err) {
        statusEl.textContent = 'Erro câmera: ' + err.message;
        return;
    }

    ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onopen = () => { 
        statusEl.textContent = 'Conectado! Streaming...'; 
        if(videoReady) sendFrame(); 
    };
    ws.onclose = () => { statusEl.textContent = 'Desconectado.'; };
    
    ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.error) {
            statusEl.textContent = 'Erro Servidor: ' + data.error;
            busy = false;
            return;
        }
        
        const img = new Image();
        img.onload = () => {
            // Volta a preencher de forma 1 pra 1 o canvas sem ter a dor de esticar a imagem
            ctx.drawImage(img, 0, 0, 640, 480);
            busy = false;
            requestAnimationFrame(sendFrame);
        };
        img.src = 'data:image/jpeg;base64,' + data.frame;

        // Mostra a predição de texto
        if (data.detections && data.detections.length > 0) {
            gestureEl.textContent = data.detections[0].gesture_name + ' (' + Math.round(data.detections[0].probability*100) + '%)';
        } else {
            gestureEl.textContent = '---';
        }

        // Atualiza FPS na tela
        if (data.fps !== undefined && valFps) {
            valFps.textContent = data.fps;
        }

        // Mostra/esconde a imagem do gesto
        if (data.image) {
            gestureImg.src = '/imagens/' + data.image;
            gestureImg.style.display = 'block';
        } else {
            gestureImg.style.display = 'none';
        }
    };
}

function sendFrame() {
    if(!ws || ws.readyState !== 1 || busy || !videoReady) return;
    busy = true;
    
    // Voltamos a sugar a Webcam integral a 640x480 (confiamos nos outros Boosts de FPS do Backend)
    offCtx.drawImage(video, 0, 0, 640, 480);
    
    let quality = 70;
    let showLandmarks = true;
    
    if (rngQuality) quality = parseInt(rngQuality.value, 10);
    if (chkLandmarks) showLandmarks = chkLandmarks.checked;

    const b64Image = off.toDataURL('image/jpeg', Math.max(0.1, quality / 100)).split(',')[1];
    
    const payload = JSON.stringify({
        image: b64Image,
        quality: quality,
        show_landmarks: showLandmarks
    });
    
    ws.send(payload);
}

start();
