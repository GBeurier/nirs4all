// Width toggle functionality
document.addEventListener('DOMContentLoaded', function() {
    // Create toggle button
    var btn = document.createElement('button');
    btn.className = 'width-toggle-btn';
    btn.innerHTML = '⇔';
    btn.title = 'Toggle wide view';
    document.body.appendChild(btn);

    // Check saved preference
    if (localStorage.getItem('wideView') === 'true') {
        document.body.classList.add('wide-view');
        btn.innerHTML = '⇿';
        btn.title = 'Toggle normal view';
    }

    // Toggle on click
    btn.addEventListener('click', function() {
        document.body.classList.toggle('wide-view');
        var isWide = document.body.classList.contains('wide-view');
        localStorage.setItem('wideView', isWide);
        btn.innerHTML = isWide ? '⇿' : '⇔';
        btn.title = isWide ? 'Toggle normal view' : 'Toggle wide view';
    });
});
