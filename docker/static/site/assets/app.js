(function () {
  "use strict";

  var apiBase = resolveApiBase();
  var pollIntervalMs = 1800;
  var maxPolls = 240;

  var nodes = {
    apiStatus: document.getElementById("apiStatus"),
    citationCount: document.getElementById("citationCount"),
    form: document.getElementById("chatForm"),
    input: document.getElementById("questionInput"),
    jobId: document.getElementById("jobId"),
    jobStatus: document.getElementById("jobStatus"),
    latencyValue: document.getElementById("latencyValue"),
    messages: document.getElementById("messages"),
    retrievalMethod: document.getElementById("retrievalMethod"),
    retrievalSelect: document.getElementById("retrievalSelect"),
    sendButton: document.getElementById("sendButton"),
    sourcesList: document.getElementById("sourcesList"),
  };

  var busy = false;
  var apiAvailable = false;
  var retrievalMethodsLoaded = false;
  var retrievalLabelsByValue = {};
  var retrievalInfoByValue = {};
  var retrievalStorageKey = "medicalRagRetrievalMethod";
  var pendingJobStorageKey = "medicalRagPendingJob";
  var resumeAttempted = false;
  var questionPlaceholder = nodes.input.getAttribute("placeholder") || "";

  function resolveApiBase() {
    var params = new URLSearchParams(window.location.search);
    var configured =
      params.get("api") ||
      window.MEDICAL_RAG_API_BASE ||
      window.localStorage.getItem("medicalRagApiBase") ||
      "/api";

    configured = String(configured).trim() || "/api";
    if (params.get("api")) {
      window.localStorage.setItem("medicalRagApiBase", configured);
    }
    return configured.replace(/\/$/, "");
  }

  function delay(ms) {
    return new Promise(function (resolve) {
      window.setTimeout(resolve, ms);
    });
  }

  function savePendingJob(jobId, question, retrievalMethod, retrievalLabel) {
    if (!jobId) {
      return;
    }
    window.localStorage.setItem(
      pendingJobStorageKey,
      JSON.stringify({
        jobId: jobId,
        question: question,
        retrievalMethod: retrievalMethod,
        retrievalLabel: retrievalLabel,
        submittedAt: new Date().toISOString(),
      }),
    );
  }

  function readPendingJob() {
    try {
      return JSON.parse(
        window.localStorage.getItem(pendingJobStorageKey) || "{}",
      );
    } catch (_error) {
      return {};
    }
  }

  function clearPendingJob(jobId) {
    var pending = readPendingJob();
    if (!jobId || pending.jobId === jobId) {
      window.localStorage.removeItem(pendingJobStorageKey);
    }
  }

  function setApiStatus(text, className) {
    nodes.apiStatus.textContent = text;
    nodes.apiStatus.className = "status-pill " + className;
  }

  function setBusy(nextBusy) {
    busy = nextBusy;
    updateComposerState();
  }

  function updateComposerState() {
    var selectedReady = selectedRetrievalReady();
    nodes.sendButton.disabled = busy || !apiAvailable || !selectedReady;
    nodes.input.disabled = busy || !apiAvailable;
    nodes.retrievalSelect.disabled = busy || !apiAvailable;
    nodes.sendButton.textContent = busy ? "Sending" : "Send";
    var placeholder = questionPlaceholder;
    if (!apiAvailable) {
      placeholder =
        "API is unavailable. Open through the nginx gateway or start the backend.";
    } else if (!selectedReady) {
      placeholder =
        selectedRetrievalInfo().message ||
        "Selected retrieval index is missing. Build the index before sending.";
    }
    nodes.input.setAttribute("placeholder", placeholder);
  }

  function setState(values) {
    nodes.jobStatus.textContent = values.status || "Idle";
    nodes.jobId.textContent = values.jobId || "None";
    nodes.citationCount.textContent = String(values.citations || 0);
    nodes.latencyValue.textContent = values.latency || "-";
    nodes.retrievalMethod.textContent =
      displayRetrievalMethod(values.method) ||
      selectedRetrievalLabel() ||
      "BM25";
  }

  function resetSources(text) {
    nodes.sourcesList.replaceChildren();
    var empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = text || "No sources selected.";
    nodes.sourcesList.appendChild(empty);
  }

  function stripHtml(text) {
    var doc = new DOMParser().parseFromString(text, "text/html");
    return (doc.body.textContent || text).replace(/\s+/g, " ").trim();
  }

  function parseJsonResponse(response) {
    return response.text().then(function (text) {
      if (!text) {
        return {};
      }
      try {
        return JSON.parse(text);
      } catch (_error) {
        return { error: stripHtml(text) };
      }
    });
  }

  function resolveErrorMessage(payload, response) {
    if (
      response.status === 501 &&
      apiBase === "/api" &&
      window.location.port === "5173"
    ) {
      return (
        "This local preview only serves static files. Open the app through " +
        "the nginx gateway, usually http://127.0.0.1:8080/, so /api can be " +
        "proxied to FastAPI."
      );
    }
    if (payload && payload.error) {
      return String(payload.error);
    }
    if (payload && payload.detail) {
      return Array.isArray(payload.detail)
        ? payload.detail.map(String).join("; ")
        : String(payload.detail);
    }
    return response.status + " " + response.statusText;
  }

  function fetchJson(path, options) {
    var requestOptions = Object.assign(
      {
        credentials: "same-origin",
        headers: { Accept: "application/json" },
      },
      options || {},
    );

    return fetch(apiBase + path, requestOptions).then(function (response) {
      return parseJsonResponse(response).then(function (payload) {
        if (!response.ok) {
          var error = new Error(resolveErrorMessage(payload, response));
          error.status = response.status;
          error.payload = payload;
          throw error;
        }
        return payload;
      });
    });
  }

  function appendMessage(role, text) {
    var item = document.createElement("li");
    item.className =
      role === "user" ? "message user-message" : "message assistant-message";

    var avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = role === "user" ? "You" : "RAG";

    var bubble = document.createElement("div");
    bubble.className = "bubble";

    if (text) {
      var paragraph = document.createElement("p");
      paragraph.textContent = text;
      bubble.appendChild(paragraph);
    }

    item.appendChild(avatar);
    item.appendChild(bubble);
    nodes.messages.appendChild(item);
    nodes.messages.scrollTop = nodes.messages.scrollHeight;
    return bubble;
  }

  function setPendingBubble(bubble, text) {
    bubble.replaceChildren();

    var row = document.createElement("div");
    row.className = "pending-row";

    var spinner = document.createElement("span");
    spinner.className = "spinner";
    spinner.setAttribute("aria-hidden", "true");

    var label = document.createElement("span");
    label.textContent = text;

    row.appendChild(spinner);
    row.appendChild(label);
    bubble.appendChild(row);
  }

  function renderError(bubble, message) {
    bubble.replaceChildren();

    var error = document.createElement("p");
    error.className = "error-box";
    error.textContent = message;
    bubble.appendChild(error);
  }

  function appendParagraphs(parent, text) {
    var normalized = String(text || "").trim();
    if (!normalized) {
      normalized = "The backend returned an empty answer.";
    }

    normalized.split(/\n{2,}/).forEach(function (chunk) {
      var paragraph = document.createElement("p");
      paragraph.textContent = chunk.replace(/\n/g, " ");
      parent.appendChild(paragraph);
    });
  }

  function formatScore(value) {
    var numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return null;
    }
    return numeric.toFixed(4);
  }

  function formatLatency(ms) {
    var numeric = Number(ms);
    if (!Number.isFinite(numeric)) {
      return null;
    }
    if (numeric >= 1000) {
      return (numeric / 1000).toFixed(1) + "s";
    }
    return Math.round(numeric) + "ms";
  }

  function readLatency(result) {
    return (
      formatLatency(result.end_to_end_latency_ms) ||
      formatLatency(result.generation_latency_ms) ||
      formatLatency(result.retrieval_latency_ms)
    );
  }

  function selectedRetrievalMethod() {
    return nodes.retrievalSelect.value || "bm25";
  }

  function selectedRetrievalLabel() {
    var option = nodes.retrievalSelect.selectedOptions[0];
    var value = selectedRetrievalMethod();
    return (
      retrievalLabelsByValue[value] || (option ? option.textContent : value)
    );
  }

  function retrievalLabelFor(method) {
    var value = String(method || "").trim();
    if (!value) {
      return "";
    }

    var options = Array.from(nodes.retrievalSelect.options);
    var match = options.find(function (option) {
      return option.value === value;
    });
    return match ? match.textContent : retrievalLabelsByValue[value] || value;
  }

  function selectedRetrievalInfo() {
    return retrievalInfoByValue[selectedRetrievalMethod()] || {};
  }

  function selectedRetrievalReady() {
    if (!retrievalMethodsLoaded) {
      return true;
    }
    var info = selectedRetrievalInfo();
    return info.index_ready !== false;
  }

  function displayRetrievalMethod(method) {
    var value = String(method || selectedRetrievalMethod()).trim();
    var label = retrievalLabelFor(value);
    var info = retrievalInfoByValue[value];
    if (info && info.index_ready === false) {
      return label + " (index missing)";
    }
    return label;
  }

  function populateRetrievalMethods(payload) {
    if (
      !payload ||
      !Array.isArray(payload.methods) ||
      payload.methods.length === 0
    ) {
      return;
    }

    var previous =
      window.localStorage.getItem(retrievalStorageKey) ||
      selectedRetrievalMethod();
    var fallback = payload.default || "bm25";
    retrievalLabelsByValue = {};
    retrievalInfoByValue = {};
    nodes.retrievalSelect.replaceChildren();

    payload.methods.forEach(function (method) {
      var value = typeof method === "string" ? method : method.value;
      var label = typeof method === "string" ? method : method.label;
      if (!value) {
        return;
      }

      var option = document.createElement("option");
      var cleanLabel = label || value;
      option.value = value;
      option.textContent =
        cleanLabel + (method.index_ready === false ? " (index missing)" : "");
      retrievalLabelsByValue[value] = cleanLabel;
      retrievalInfoByValue[value] =
        typeof method === "string" ? { index_ready: true } : method;
      nodes.retrievalSelect.appendChild(option);
    });

    var values = Array.from(nodes.retrievalSelect.options).map(
      function (option) {
        return option.value;
      },
    );
    nodes.retrievalSelect.value =
      values.indexOf(previous) >= 0 ? previous : fallback;
    window.localStorage.setItem(retrievalStorageKey, selectedRetrievalMethod());
    setState({ method: selectedRetrievalMethod() });
    updateComposerState();
  }

  function renderMeta(bubble, result) {
    var meta = document.createElement("div");
    meta.className = "answer-meta";

    var items = [];
    if (Array.isArray(result.retrieved)) {
      items.push(result.retrieved.length + " sources");
    }
    if (typeof result.reranker_enabled === "boolean") {
      items.push(result.reranker_enabled ? "Reranker on" : "Reranker off");
    }
    if (result.retrieval_method) {
      items.push("Retrieval " + retrievalLabelFor(result.retrieval_method));
    }
    var latency = readLatency(result);
    if (latency) {
      items.push("Latency " + latency);
    }

    items.forEach(function (label) {
      var chip = document.createElement("span");
      chip.className = "meta-chip";
      chip.textContent = label;
      meta.appendChild(chip);
    });

    if (items.length > 0) {
      bubble.appendChild(meta);
    }
  }

  function renderCitations(bubble, result) {
    var citations = Array.isArray(result.citations_detected)
      ? result.citations_detected
      : [];
    if (citations.length === 0) {
      return;
    }

    var supported = new Set(result.supported_citations_detected || []);
    var unsupported = new Set(result.unsupported_citations_detected || []);
    var row = document.createElement("div");
    row.className = "citation-row";

    citations.forEach(function (citation) {
      var chip = document.createElement("span");
      chip.className = "citation-chip";
      if (supported.has(citation)) {
        chip.classList.add("supported");
      }
      if (unsupported.has(citation)) {
        chip.classList.add("unsupported");
      }
      chip.textContent = "[" + citation + "]";
      row.appendChild(chip);
    });

    bubble.appendChild(row);
  }

  function renderSources(result) {
    var sources = Array.isArray(result.retrieved) ? result.retrieved : [];
    nodes.sourcesList.replaceChildren();

    if (sources.length === 0) {
      resetSources("No retrieved sources returned.");
      return;
    }

    sources.forEach(function (source, index) {
      var details = document.createElement("details");
      details.className = "source-item";
      if (index === 0) {
        details.open = true;
      }

      var summary = document.createElement("summary");
      var rank = document.createElement("span");
      rank.className = "source-rank";
      rank.textContent = String(index + 1);

      var title = document.createElement("span");
      title.className = "source-title";

      var id = document.createElement("span");
      id.className = "source-id";
      id.textContent = source.doc_id || "Unknown document";

      var meta = document.createElement("span");
      meta.className = "source-meta";
      var parts = [];
      var score = formatScore(source.score);
      var retrievalScore = formatScore(source.retrieval_score);
      var rerankerScore = formatScore(source.reranker_score);

      if (source.source) {
        parts.push(source.source);
      }
      if (score) {
        parts.push("score " + score);
      }
      if (retrievalScore && rerankerScore) {
        parts.push("retrieval " + retrievalScore);
        parts.push("rerank " + rerankerScore);
      }
      meta.textContent = parts.join(" | ");

      title.appendChild(id);
      title.appendChild(meta);
      summary.appendChild(rank);
      summary.appendChild(title);

      var text = document.createElement("p");
      text.className = "source-text";
      text.textContent = source.text || "No passage text returned.";

      details.appendChild(summary);
      details.appendChild(text);
      nodes.sourcesList.appendChild(details);
    });
  }

  function renderResult(bubble, result, job) {
    bubble.replaceChildren();
    appendParagraphs(bubble, result.answer);
    renderCitations(bubble, result);
    renderMeta(bubble, result);
    renderSources(result);
    if (job && job.job_id) {
      clearPendingJob(job.job_id);
    }

    setState({
      status: job && job.status ? job.status : "SUCCEEDED",
      jobId: job && job.job_id ? job.job_id : "None",
      citations: Array.isArray(result.citations_detected)
        ? result.citations_detected.length
        : 0,
      latency: readLatency(result) || "-",
      method:
        result.retrieval_method ||
        (job && job.metadata && job.metadata.retrieval_method) ||
        selectedRetrievalLabel(),
    });
  }

  function pollJob(jobId, bubble) {
    var encodedJobId = encodeURIComponent(jobId);
    var polls = 0;

    function tick() {
      polls += 1;
      return fetchJson("/jobs/" + encodedJobId).then(function (job) {
        setState({
          status: job.status,
          jobId: job.job_id,
          citations: 0,
          latency: "-",
          method:
            (job.metadata && job.metadata.retrieval_method) ||
            selectedRetrievalLabel(),
        });

        if (job.status === "FAILED") {
          var failed = new Error(job.error_message || "Job failed.");
          failed.job = job;
          clearPendingJob(job.job_id);
          throw failed;
        }

        if (job.status === "SUCCEEDED") {
          return fetchJson("/jobs/" + encodedJobId + "/result").then(
            function (result) {
              renderResult(bubble, result, job);
            },
          );
        }

        if (polls >= maxPolls) {
          throw new Error("Timed out waiting for the answer.");
        }

        setPendingBubble(
          bubble,
          job.status === "RUNNING" ? "Generating answer" : "Waiting in queue",
        );
        return delay(pollIntervalMs).then(tick);
      });
    }

    return delay(pollIntervalMs).then(tick);
  }

  function submitQuestion(question) {
    var retrievalMethod = selectedRetrievalMethod();
    var retrievalLabel = selectedRetrievalLabel();
    appendMessage("user", question);
    var assistantBubble = appendMessage("assistant");
    setPendingBubble(assistantBubble, "Submitting question");
    resetSources("Waiting for retrieval.");
    setState({
      status: "Submitting",
      jobId: "None",
      citations: 0,
      latency: "-",
      method: retrievalMethod,
    });
    setBusy(true);

    return fetchJson("/jobs", {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: question,
        retrieval_method: retrievalMethod,
        metadata: {
          source: "static.chat_ui",
          retrieval_method: retrievalMethod,
          submitted_at: new Date().toISOString(),
        },
      }),
    })
      .then(function (payload) {
        var jobId = payload.job_id;
        savePendingJob(jobId, question, retrievalMethod, retrievalLabel);
        setState({
          status: payload.status || "PENDING",
          jobId: jobId || "None",
          citations: 0,
          latency: "-",
          method:
            (payload.metadata && payload.metadata.retrieval_method) ||
            retrievalMethod,
        });

        if (payload.result) {
          renderResult(assistantBubble, payload.result, payload);
          return null;
        }

        if (!jobId) {
          throw new Error("Backend did not return a job id.");
        }

        return pollJob(jobId, assistantBubble);
      })
      .catch(function (error) {
        renderError(assistantBubble, error.message || "Request failed.");
        resetSources("No sources available for this request.");
        var failedJob = error.job || error.payload || {};
        if (failedJob.status === "FAILED") {
          clearPendingJob(failedJob.job_id);
        }
        setState({
          status: "Error",
          jobId: failedJob.job_id || "None",
          citations: 0,
          latency: "-",
          method:
            (failedJob.metadata && failedJob.metadata.retrieval_method) ||
            retrievalMethod,
        });
      })
      .finally(function () {
        setBusy(false);
        nodes.input.focus();
      });
  }

  function resizeInput() {
    nodes.input.style.height = "auto";
    nodes.input.style.height = nodes.input.scrollHeight + "px";
  }

  function bindEvents() {
    nodes.form.addEventListener("submit", function (event) {
      event.preventDefault();
      if (busy) {
        return;
      }

      var question = nodes.input.value.trim();
      if (!question) {
        nodes.input.focus();
        return;
      }
      if (!selectedRetrievalReady()) {
        setState({ method: selectedRetrievalMethod() });
        nodes.retrievalSelect.focus();
        return;
      }

      nodes.input.value = "";
      resizeInput();
      submitQuestion(question);
    });

    nodes.input.addEventListener("input", resizeInput);

    nodes.input.addEventListener("keydown", function (event) {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        nodes.form.requestSubmit();
      }
    });

    document.querySelectorAll("[data-prompt]").forEach(function (button) {
      button.addEventListener("click", function () {
        nodes.input.value = button.getAttribute("data-prompt") || "";
        resizeInput();
        nodes.input.focus();
      });
    });

    nodes.retrievalSelect.addEventListener("change", function () {
      window.localStorage.setItem(
        retrievalStorageKey,
        selectedRetrievalMethod(),
      );
      setState({ method: selectedRetrievalMethod() });
      updateComposerState();
    });
  }

  function checkHealth() {
    fetchJson("/health")
      .then(function () {
        apiAvailable = true;
        setApiStatus("API online", "status-online");
        updateComposerState();
        if (!retrievalMethodsLoaded) {
          fetchJson("/retrieval-methods")
            .then(function (payload) {
              retrievalMethodsLoaded = true;
              populateRetrievalMethods(payload);
              resumeLastPendingJob();
            })
            .catch(function () {
              retrievalMethodsLoaded = true;
              updateComposerState();
              resumeLastPendingJob();
            });
        } else {
          resumeLastPendingJob();
        }
      })
      .catch(function () {
        apiAvailable = false;
        setApiStatus("API unavailable", "status-error");
        updateComposerState();
      });
  }

  function resumeLastPendingJob() {
    if (resumeAttempted || busy || !apiAvailable) {
      return;
    }
    resumeAttempted = true;

    var pending = readPendingJob();
    if (!pending.jobId) {
      return;
    }

    fetchJson("/jobs/" + encodeURIComponent(pending.jobId))
      .then(function (job) {
        if (job.status === "FAILED") {
          clearPendingJob(job.job_id);
          return null;
        }

        appendMessage("user", pending.question || "Previous question");
        var assistantBubble = appendMessage("assistant");
        setPendingBubble(assistantBubble, "Resuming answer");
        resetSources("Resuming retrieval.");
        setState({
          status: job.status || "PENDING",
          jobId: job.job_id,
          citations: 0,
          latency: "-",
          method:
            (job.metadata && job.metadata.retrieval_method) ||
            pending.retrievalLabel ||
            selectedRetrievalLabel(),
        });

        setBusy(true);
        return pollJob(job.job_id, assistantBubble).finally(function () {
          setBusy(false);
          nodes.input.focus();
        });
      })
      .catch(function () {
        clearPendingJob(pending.jobId);
      });
  }

  bindEvents();
  resetSources();
  resizeInput();
  setState({ method: selectedRetrievalMethod() });
  updateComposerState();
  checkHealth();
  window.setInterval(checkHealth, 10000);
})();
