let currentLabel = 'A';
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Handle class selection
function setLabel(label) {
  currentLabel = label;
}

// Handle canvas click to add point
canvas.addEventListener('click', async (e) => {
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  // Draw the point
  drawDot(x, y, currentLabel);

  // Send point to server
  await fetch('/add_point', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ x: x, y: y, label: currentLabel })
  });
});

// Draw a dot on canvas
function drawDot(x, y, label) {
  ctx.fillStyle = label === 'A' ? 'red' : label === 'B' ? 'green' : 'blue';
  ctx.beginPath();
  ctx.arc(x, y, 5, 0, 2 * Math.PI);
  ctx.fill();
}

// Train the model
async function trainModel() {
  const response = await fetch('/train', { method: 'POST' });
  alert("Model trained!");
}

// Show loss chart
async function showLoss() {
    const res = await fetch('/get_loss');
    const data = await res.json();
  
    if (!data.loss || data.loss.length === 0) {
      alert("No loss data received!");
      return;
    }
  
    console.log("LOSS DATA:", data.loss);  // debug
  
    const ctxChart = document.getElementById('lossChart').getContext('2d');
  
    new Chart(ctxChart, {
      type: 'line',
      data: {
        labels: data.loss.map((_, i) => i),
        datasets: [{
          label: 'Loss',
          data: data.loss,
          borderColor: 'orange',
          fill: false,
          tension: 0.1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
      }
    });
  }
  

// Visualize prediction
async function predictCanvas() {
  const res = await fetch('/predict_canvas');
  const data = await res.json();
  const prediction = data.predicted;

  const image = ctx.createImageData(500, 500);
  const colorMap = [
    [255, 100, 100], // red for A
    [100, 255, 100], // green for B
    [100, 100, 255]  // blue for C
  ];

  for (let y = 0; y < 500; y++) {
    for (let x = 0; x < 500; x++) {
      const idx = (y * 500 + x) * 4;
      const color = colorMap[prediction[y][x]];
      image.data[idx] = color[0];
      image.data[idx + 1] = color[1];
      image.data[idx + 2] = color[2];
      image.data[idx + 3] = 255;
    }
  }

  ctx.putImageData(image, 0, 0);
}
