// Word counter, busy state, keyboard submit, and "Edit the text" / "Restore your text".
// Nothing here formats a probability: every number on the page comes from utils/present.py.
(() => {
  const form = document.getElementById('analyse-form');
  const text = document.getElementById('text');
  const count = document.getElementById('count');
  const button = document.getElementById('analyse');
  if (!form || !text || !count || !button) return;

  const DRAFT_KEY = 'fnd-draft';
  const thousands = new Intl.NumberFormat('en-US');
  // The server splits with Python's str.split(); these characters are treated as spaces on both sides.
  const words = (s) => s.replace(/[\u001c-\u001f\u0085\ufeff]/g, ' ').split(/\s+/).filter(Boolean).length;
  const setCount = () => {
    const n = words(text.value);
    count.textContent = `${thousands.format(n)} ${n === 1 ? 'word' : 'words'}`;
  };

  let busy = false;
  const setBusy = (on) => {
    busy = on;
    button.textContent = on ? 'Analysing…' : 'Analyse';
    button.classList.toggle('busy', on);
    if (on) button.setAttribute('aria-disabled', 'true');
    else button.removeAttribute('aria-disabled');
  };

  form.addEventListener('submit', (event) => {
    // Never `disabled`: a disabled control is left out of the submitted form.
    if (busy) { event.preventDefault(); return; }
    try { sessionStorage.setItem(DRAFT_KEY, text.value); } catch (e) { /* storage may be unavailable */ }
    setBusy(true);
  });

  // Back/forward cache restores the page as it was; don't leave it stuck on "Analysing…".
  window.addEventListener('pageshow', (event) => { if (event.persisted) setBusy(false); });

  text.addEventListener('input', setCount);
  text.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  document.querySelectorAll('[data-edit]').forEach((link) => {
    link.addEventListener('click', (event) => {
      event.preventDefault();
      text.focus();
      const end = text.value.length;
      text.setSelectionRange(end, end);
    });
  });

  const restore = document.querySelector('[data-restore]');
  if (restore) {
    let draft = null;
    try { draft = sessionStorage.getItem(DRAFT_KEY); } catch (e) { draft = null; }
    if (draft) {
      restore.hidden = false;
      restore.addEventListener('click', () => {
        text.value = draft;
        setCount();
        text.focus();
      });
    }
  }

  setCount();
})();
