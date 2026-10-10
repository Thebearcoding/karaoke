// Apply before the page paints; a saved choice takes precedence over the OS.
(() => {
  const key = 'karaokeTheme';
  const system = matchMedia('(prefers-color-scheme: dark)');
  const read = () => {
    try {
      const value = localStorage.getItem(key);
      return value === 'dark' || value === 'light' ? value : null;
    } catch { return null; }
  };
  let preference = read();
  function apply() {
    const dark = preference ? preference === 'dark' : system.matches;
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
    document.querySelector('meta[name="theme-color"]').content = dark ? '#202022' : '#ffffff';
    const button = document.getElementById('theme-toggle');
    if (button) {
      button.title = dark ? '切换到日间模式' : '切换到夜间模式';
      button.setAttribute('aria-label', button.title);
    }
  }
  apply();
  document.addEventListener('DOMContentLoaded', () => {
    apply();
    document.getElementById('theme-toggle').addEventListener('click', () => {
      preference = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      try { localStorage.setItem(key, preference); } catch {}
      apply();
    });
  }, {once:true});
  system.addEventListener('change', apply);
  window.addEventListener('storage', event => {
    if (event.key === key || event.key === null) { preference = read(); apply(); }
  });
})();
