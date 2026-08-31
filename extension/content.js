// Ensure we only inject the container once
if (!window.imageSignalInjected) {
  window.imageSignalInjected = true;

  const container = document.createElement('div');
  container.id = 'imagesignal-overlay-container';
  document.body.appendChild(container);

  let activeToast = null;
  let activeToastId = null;

  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'show_loading') {
      showLoadingToast();
    } else if (request.action === 'show_result') {
      showResultToast(request.result);
    } else if (request.action === 'show_error') {
      showErrorToast(request.error);
    }
  });

  function showLoadingToast() {
    if (activeToast) activeToast.remove();
    activeToastId = 'toast-' + Math.random().toString(36).substr(2, 9);
    
    activeToast = document.createElement('div');
    activeToast.className = 'is-toast';
    activeToast.id = activeToastId;
    
    activeToast.innerHTML = `
      <div class="is-toast-header">
        <div class="is-toast-brand">ImageSignal</div>
        <button class="is-toast-close">&times;</button>
      </div>
      <div class="is-toast-content">
        <div class="is-loading">
          <div class="is-spinner"></div>
          <span>Analyzing image...</span>
        </div>
      </div>
    `;
    
    container.appendChild(activeToast);
    
    activeToast.querySelector('.is-toast-close').addEventListener('click', () => {
      activeToast.remove();
      activeToast = null;
    });
  }

  function showResultToast(result) {
    if (!activeToast) return;
    
    let verdict = '';
    let tone = '';
    const score = result.reencoded_score;
    
    if (score >= 0.7) {
      verdict = 'Likely AI-Generated';
      tone = 'high';
    } else if (score >= 0.4) {
      verdict = 'Inconclusive';
      tone = 'mid';
    } else {
      verdict = 'Likely Authentic';
      tone = 'low';
    }

    const content = activeToast.querySelector('.is-toast-content');
    content.innerHTML = `
      <div class="is-result-banner ${tone}">
        <div>
          <h4 class="is-verdict">${verdict}</h4>
          <p class="is-score">AI Signal Score: ${(score * 100).toFixed(1)}%</p>
        </div>
      </div>
    `;

    scheduleRemoval();
  }

  function showErrorToast(errorMsg) {
    if (!activeToast) return;
    const content = activeToast.querySelector('.is-toast-content');
    content.innerHTML = `
      <div class="is-error">
        Failed to analyze image. Ensure your local server (localhost:8000) is running.
      </div>
    `;
    scheduleRemoval(5000);
  }

  function scheduleRemoval(ms = 10000) {
    const toastToRemove = activeToast;
    setTimeout(() => {
      if (toastToRemove && document.body.contains(toastToRemove)) {
        toastToRemove.style.opacity = '0';
        toastToRemove.style.transform = 'translateY(20px)';
        toastToRemove.style.transition = 'all 0.3s ease';
        setTimeout(() => {
          if (toastToRemove && document.body.contains(toastToRemove)) {
            toastToRemove.remove();
          }
        }, 300);
      }
    }, ms);
  }
}

