const target = document.getElementById('target');
const scoreDisplay = document.getElementById('score');
const highScoreDisplay = document.getElementById('high-score');
const timerDisplay = document.getElementById('timer');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const resetBtn = document.getElementById('reset-btn');

let score = 0;
let highScore = localStorage.getItem('highScore') || 0;
let timeLeft = 30;
let gameInterval;
let targetTimeout;
let isPlaying = false;

highScoreDisplay.textContent = highScore;

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    document.getElementById('webcam').srcObject = stream;
  });

  function getRandomPosition() {
    const targetSize = 60;
  
    const zones = [
      {
        // Center zone
        minX: window.innerWidth * 0.25,
        maxX: window.innerWidth * 0.75 - targetSize,
        minY: window.innerHeight * 0.25,
        maxY: window.innerHeight * 0.75 - targetSize,
      },
      {
        // Top-right corner
        minX: window.innerWidth * 0.75,
        maxX: window.innerWidth - targetSize,
        minY: 80, // Avoid overlapping title
        maxY: window.innerHeight * 0.25 - targetSize,
      },
      {
        // Bottom-left corner
        minX: 0,
        maxX: window.innerWidth * 0.25 - targetSize,
        minY: window.innerHeight * 0.75,
        maxY: window.innerHeight - targetSize,
      },
    ];
  
    // Choose a random zone
    const zone = zones[Math.floor(Math.random() * zones.length)];
  
    const x = Math.random() * (zone.maxX - zone.minX) + zone.minX;
    const y = Math.random() * (zone.maxY - zone.minY) + zone.minY;
  
    return { x, y };
  }
  
  
function showTarget() {
  if (!isPlaying) return;
  const { x, y } = getRandomPosition();
  target.style.left = `${x}px`;
  target.style.top = `${y}px`;
  target.style.display = 'block';

  targetTimeout = setTimeout(() => {
    target.style.display = 'none';
    showTarget();
  }, 1500);
}

function updateScore() {
  score++;
  scoreDisplay.textContent = score;
  if (score > highScore) {
    highScore = score;
    highScoreDisplay.textContent = highScore;
    localStorage.setItem('highScore', highScore);
  }
}

target.addEventListener('click', () => {
  const sound = new Audio('pop.mp3');
  sound.play();
  updateScore();
  clearTimeout(targetTimeout);
  target.style.display = 'none';
  setTimeout(showTarget, 500);
});


function startGame() {
  if (isPlaying) return;
  isPlaying = true;

  
 
  score = 0;
  scoreDisplay.textContent = score;
  timeLeft = 30;
  timerDisplay.textContent = timeLeft;
  showTarget();

  gameInterval = setInterval(() => {
    timeLeft--;
    timerDisplay.textContent = timeLeft;
    if (timeLeft <= 0) {
      stopGame();
      alert('Time’s up! Your score: ' + score);
    }
  }, 1000);
}
resetBtn.addEventListener('click', () => {
  if (confirm("Are you sure you want to reset your high score?")) {
    highScore = 0;
    localStorage.setItem('highScore', 0);
    highScoreDisplay.textContent = 0;
  }
});


function stopGame() {
  isPlaying = false;
  clearInterval(gameInterval);
  clearTimeout(targetTimeout);
  target.style.display = 'none';
}

startBtn.addEventListener('click', startGame);
stopBtn.addEventListener('click', stopGame);
