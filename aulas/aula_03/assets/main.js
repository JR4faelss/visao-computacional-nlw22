const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const gestureEl = document.getElementById('gesture_text');
const gestureImg = document.getElementById('gesture_img');

// Câmera escondida para captura
const video = document.createElement('video');
video.setAttribute('playsinline', '');
video.setAttribute('autoplay', '');
video.style.opacity = '0';
video.style.position = 'absolute';
document.body.appendChild(video);

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
            ctx.drawImage(img, 0, 0);
            busy = false;
            requestAnimationFrame(sendFrame);
        };
        img.src = 'data:image/jpeg;base64,' + data.frame;

        if (data.detections && data.detections.length > 0) {
            gestureEl.textContent = data.detections[0].gesture_name + ' (' + Math.round(data.detections[0].probability*100) + '%)';
        } else {
            gestureEl.textContent = '---';
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
    offCtx.drawImage(video, 0, 0, 640, 480);
    ws.send(off.toDataURL('image/jpeg', 0.6).split(',')[1]);
}

start();
