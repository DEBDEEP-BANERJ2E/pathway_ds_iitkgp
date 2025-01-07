document.addEventListener("DOMContentLoaded", function () {
    const uploadForm = document.querySelector("form");
    const fileInput = document.getElementById("paper");

    uploadForm.addEventListener("submit", function (event) {
        event.preventDefault();

        const formData = new FormData();
        formData.append("paper", fileInput.files[0]);

        fetch("/upload", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            window.location.href = "/results";  // Redirect to results page after upload
        })
        .catch(error => {
            alert("Error uploading paper. Please try again.");
        });
    });
});
