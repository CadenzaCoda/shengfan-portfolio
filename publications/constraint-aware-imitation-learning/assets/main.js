'use strict';

// These controls only explain the architecture; they do not simulate a policy.
const board = document.querySelector('.method-board');
const stageNote = document.querySelector('#stage-note');
document.querySelectorAll('[data-stage]').forEach(button => {
  button.addEventListener('click', () => {
    const stage = button.dataset.stage;
    board.dataset.mode = stage;
    document.querySelectorAll('[data-stage]').forEach(item => {
      const selected = item === button;
      item.classList.toggle('active', selected);
      item.setAttribute('aria-pressed', String(selected));
    });
    stageNote.textContent = stage === 'deployment'
      ? 'Deployment: observations go directly through the learned policy to throttle and steering. The faded supervision modules are not used to generate actions.'
      : 'Training: the expert and learned safety models provide supervision. Rollouts and expert labels are collected iteratively.';
  });
});

document.querySelector('#load-video').addEventListener('click', () => {
  const frame = document.createElement('iframe');
  frame.src = 'https://www.youtube-nocookie.com/embed/fyiEeHxZiMM?autoplay=1';
  frame.title = 'CAIL: Constraint-Aware Imitation Learning — project video';
  frame.allow = 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share';
  frame.allowFullscreen = true;
  frame.referrerPolicy = 'strict-origin-when-cross-origin';
  document.querySelector('#video-shell').replaceChildren(frame);
  frame.focus();
});

document.querySelector('#copy-citation').addEventListener('click', async () => {
  const status = document.querySelector('#copy-status');
  const code = document.querySelector('#bibtex');
  try {
    await navigator.clipboard.writeText(code.textContent);
    status.textContent = 'BibTeX copied.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(code);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});
