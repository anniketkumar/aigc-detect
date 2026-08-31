chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "analyze-imagesignal",
    title: "Analyze with ImageSignal",
    contexts: ["image"]
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "analyze-imagesignal") {
    
    // 1. Inject the UI container first
    try {
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ["content.js"]
      });
      await chrome.scripting.insertCSS({
        target: { tabId: tab.id },
        files: ["content.css"]
      });
    } catch (err) {
      console.error("Script injection failed: ", err);
      return;
    }

    // 2. Tell the content script to show the loading UI
    chrome.tabs.sendMessage(tab.id, { action: "show_loading" });

    // 3. Do the fetching in the background script to bypass page CSP and CORS
    try {
      // Fetch the image
      const imageResponse = await fetch(info.srcUrl);
      const blob = await imageResponse.blob();

      // Send to local API
      const formData = new FormData();
      formData.append('image', blob, 'image.jpg');
      formData.append('checkpoint', 'aug');
      formData.append('quality', '95');
      formData.append('fast_mode', 'true');

      const apiResponse = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        body: formData
      });

      if (!apiResponse.ok) {
        throw new Error('Analysis API failed');
      }

      const result = await apiResponse.json();
      
      // 4. Send the result back to the content script
      chrome.tabs.sendMessage(tab.id, { 
        action: "show_result", 
        result: result 
      });

    } catch (err) {
      console.error("Background processing error:", err);
      chrome.tabs.sendMessage(tab.id, { 
        action: "show_error", 
        error: err.message || "Failed to analyze image." 
      });
    }
  }
});

