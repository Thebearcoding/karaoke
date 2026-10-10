// Apply before the page paints; remember a choice, otherwise start in night mode.
(() => {
  const key = 'karaokeTheme';
  const read = () => {
    try {
      const value = localStorage.getItem(key);
      return value === 'light' ? 'light' : 'dark';
    } catch { return 'dark'; }
  };
  let preference = read();
  function apply() {
    const dark = preference === 'dark';
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
  window.addEventListener('storage', event => {
    if (event.key === key || event.key === null) { preference = read(); apply(); }
  });
})();
