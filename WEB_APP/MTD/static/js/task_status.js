// static/js/task_status.js
document.addEventListener('DOMContentLoaded', function() {
    const logOutput = document.querySelector('.log-output');
    const progress = document.querySelector('.progress-bar');

    function updateStatus() {
        fetch(`/api/task/${taskId}/`)
            .then(response => response.json())
            .then(data => {
                progress.style.width = `${data.progress}%`;
                logOutput.innerHTML = data.log.join('\n');
            })
            .catch(error => console.error('Error:', error));
    }

    // 每2秒轮询更新
    setInterval(updateStatus, 2000);
});
