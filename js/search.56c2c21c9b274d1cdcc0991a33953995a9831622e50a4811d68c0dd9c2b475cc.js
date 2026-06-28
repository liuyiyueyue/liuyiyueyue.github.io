(() => {
  // <stdin>
  (function() {
    const config = window.blogSearch || {};
    const input = document.getElementById("search-input");
    const status = document.getElementById("search-status");
    const results = document.getElementById("search-results");
    if (!input || !status || !results) {
      return;
    }
    let fuse = null;
    let pages = [];
    function setStatus(message) {
      status.textContent = message;
    }
    function clearResults() {
      results.replaceChildren();
    }
    function renderResults(matches) {
      clearResults();
      if (matches.length === 0) {
        setStatus(config.noResultsText || "No matching posts found.");
        return;
      }
      setStatus("");
      const fragment = document.createDocumentFragment();
      matches.slice(0, 20).forEach(({ item }) => {
        const link = document.createElement("a");
        const title = document.createElement("strong");
        const summary = document.createElement("p");
        link.className = "search-result";
        link.href = item.url;
        title.textContent = item.title;
        summary.textContent = item.summary || "";
        link.append(title, summary);
        fragment.append(link);
      });
      results.append(fragment);
    }
    function runSearch() {
      const query = input.value.trim();
      if (!query) {
        clearResults();
        setStatus(config.emptyText || "Type to search posts.");
        return;
      }
      if (!fuse) {
        return;
      }
      renderResults(fuse.search(query));
    }
    setStatus(config.loadingText || "Loading search index...");
    function fetchIndex(indexURL) {
      return fetch(indexURL).then((response) => {
        if (!response.ok) {
          throw new Error("Search index request failed.");
        }
        return response.json();
      });
    }
    fetchIndex(config.indexURL || "/index.json").then((response) => {
      if (Array.isArray(response) && response.length === 0 && config.indexURL !== "/index.json") {
        return fetchIndex("/index.json");
      }
      return response;
    }).then((data) => {
      pages = Array.isArray(data) ? data : [];
      fuse = new Fuse(pages, {
        keys: [
          { name: "title", weight: 0.7 },
          { name: "summary", weight: 0.2 },
          { name: "content", weight: 0.1 }
        ],
        threshold: 0.35,
        ignoreLocation: true,
        minMatchCharLength: 2
      });
      input.disabled = false;
      runSearch();
    }).catch(() => {
      input.disabled = true;
      clearResults();
      setStatus(config.errorText || "Search index failed to load.");
    });
    input.addEventListener("input", runSearch);
  })();
})();
