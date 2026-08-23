(function () {
  var API_URL = "http://127.0.0.1:8000/chat";
  var USER_ID = document.currentScript.getAttribute("data-user-id") || ("user_" + Math.random().toString(36).slice(2, 10));

  var bubble = document.createElement("div");
  bubble.innerHTML = "\u{1F4AC}";
  Object.assign(bubble.style, {
    position: "fixed", bottom: "20px", right: "20px",
    width: "56px", height: "56px", borderRadius: "50%",
    background: "#2563eb", color: "white", display: "flex",
    alignItems: "center", justifyContent: "center",
    fontSize: "24px", cursor: "pointer", boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
    zIndex: "999999"
  });
  document.body.appendChild(bubble);

  var panel = document.createElement("div");
  Object.assign(panel.style, {
    position: "fixed", bottom: "88px", right: "20px",
    width: "360px", height: "520px", background: "white",
    borderRadius: "14px", boxShadow: "0 8px 30px rgba(0,0,0,0.25)",
    display: "none", flexDirection: "column", overflow: "hidden",
    fontFamily: "-apple-system, sans-serif", zIndex: "999999"
  });
  panel.innerHTML = '<div style="background:#1f2937;color:white;padding:14px;font-weight:600;font-size:15px;">MemoOS</div><div id="memoos-messages" style="flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:8px;background:#f7f7f8;"></div><div style="display:flex;gap:6px;padding:10px;border-top:1px solid #e5e7eb;background:white;"><input id="memoos-input" type="text" placeholder="Type a message..." style="flex:1;padding:10px 12px;border:1px solid #d1d5db;border-radius:18px;font-size:14px;outline:none;"><button id="memoos-send" style="background:#2563eb;color:white;border:none;border-radius:18px;padding:0 16px;font-size:14px;cursor:pointer;">Send</button></div>';
  document.body.appendChild(panel);

  bubble.addEventListener("click", function () {
    panel.style.display = panel.style.display === "none" ? "flex" : "none";
  });

  var messagesEl = panel.querySelector("#memoos-messages");
  var inputEl = panel.querySelector("#memoos-input");
  var sendBtn = panel.querySelector("#memoos-send");

  function addMessage(text, sender) {
    var div = document.createElement("div");
    div.textContent = text;
    Object.assign(div.style, {
      maxWidth: "80%", padding: "9px 12px", borderRadius: "12px",
      fontSize: "13.5px", lineHeight: "1.4", whiteSpace: "pre-wrap",
      alignSelf: sender === "user" ? "flex-end" : "flex-start",
      background: sender === "user" ? "#2563eb" : "white",
      color: sender === "user" ? "white" : "#111827",
      border: sender === "user" ? "none" : "1px solid #e5e7eb"
    });
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  var isSending = false;

  function sendMessage() {
    var text = inputEl.value.trim();
    if (!text || isSending) return;

    isSending = true;
    sendBtn.disabled = true;
    addMessage(text, "user");
    inputEl.value = "";
    var loadingEl = addMessage("Thinking...", "bot");

    var controller = new AbortController();
    var timeout = setTimeout(function () { controller.abort(); }, 20000);

    fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: USER_ID, message: text }),
      signal: controller.signal
    }).then(function (res) {
      if (!res.ok) {
        throw new Error("Server returned " + res.status);
      }
      return res.json();
    }).then(function (data) {
      loadingEl.textContent = data.reply || "Sorry, I didn't get a response - please try again.";
    }).catch(function (e) {
      if (e.name === "AbortError") {
        loadingEl.textContent = "That's taking longer than expected - please try again.";
      } else {
        loadingEl.textContent = "Sorry, I'm having trouble connecting right now. Please try again in a moment.";
      }
    }).finally(function () {
      clearTimeout(timeout);
      isSending = false;
      sendBtn.disabled = false;
      inputEl.focus();
    });
  }

  sendBtn.addEventListener("click", sendMessage);
  inputEl.addEventListener("keydown", function (e) { if (e.key === "Enter") sendMessage(); });

  addMessage("Hi! I'm an AI with memory - tell me things, ask me anything, and I'll remember for next time.", "bot");
})();
