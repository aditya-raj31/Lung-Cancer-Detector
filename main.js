const form = document.querySelector('form');
form.addEventListener('submit', (e) => {
	e.preventDefault();
	const file = document.querySelector('input[type=file]').files[0];
	const formData = new FormData();
	formData.append('file', file);
	fetch('/predict', {
		method: 'POST',
		body: formData
	})
	.then(response => response.json())
	.then(data => {
		console.log(data.prediction);
		// display prediction output
	})
	.catch(error => console.error(error));
});