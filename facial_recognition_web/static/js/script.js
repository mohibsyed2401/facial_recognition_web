document.addEventListener("DOMContentLoaded", () => {
  const progressContainer = document.getElementById("progress-container");
  const progressBar = document.getElementById("progress-bar");

  // This function listens for key presses (like 'c') and updates the progress bar
  document.addEventListener("keydown", (e) => {
    if (e.key === "c" || e.key === "C") {
      // Show progress bar if hidden
      if (progressContainer.style.display === "none" || progressContainer.style.display === "") {
        progressContainer.style.display = "block";
      }

      // Get current progress
      let currentPercent = parseInt(progressBar.style.width) || 0;
      if (currentPercent < 100) {
        currentPercent += 5; // 20 captures => 5% each
        progressBar.style.width = currentPercent + "%";
        progressBar.textContent = currentPercent + "%";
      }

      // When complete, reset after a small delay
      if (currentPercent >= 100) {
        setTimeout(() => {
          progressBar.style.width = "0%";
          progressBar.textContent = "0%";
          progressContainer.style.display = "none";
        }, 2000);
      }
    }
  });
});
