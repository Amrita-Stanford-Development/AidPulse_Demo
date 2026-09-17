(function () {
  function switchTab(name) {
    document.querySelectorAll('.tabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
    document.getElementById('tab-cardiology').hidden = name !== 'cardiology';
    document.getElementById('tab-ophthalmology').hidden = name !== 'ophthalmology';
  }

  window.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.tabs button').forEach((b) => {
      b.addEventListener('click', () => switchTab(b.dataset.tab));
    });
  });
})();
