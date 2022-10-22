// Get the form and file field
let form = document.querySelector("#upload");
let file = document.querySelector("#file");
let app = document.querySelector("#app");
let pred = document.querySelector("#prediction");
let load = document.querySelector("#loading");

function logFile(event) {
  load.classList.remove("hidden");
  load.classList.add("shown");
  let str = event.target.result;
  let img = document.createElement("img");
  img.src = str;
  app.innerHTML = "";
  app.append(img);
  let req = new XMLHttpRequest();
  let formData = new FormData();
  let params = {
    image_name: ["ISIC_0251455"],
    patient_id: ["IP_3579794"],
    sex: ["male"],
    age_approx: [50.0],
    anatom_site_general_challenge: ["torso"],
    width: [6000],
    height: [4000],
  };
  formData.append("params", JSON.stringify(params));
  formData.append("photo", str);

  req.onreadystatechange = function () {
    if (req.readyState == XMLHttpRequest.DONE) {
      load.classList.remove("shown");
      load.classList.add("hidden");
      resp = req.responseText;
      pred.innerHTML = resp;
      document.getElementById("prediction").hidden = false;
    }
  };
  req.open("POST", "/predict/input");
  req.send(formData);
}

function handleSubmit(event) {
  event.preventDefault();
  // If there's no file, do nothing
  if (!file.value.length) return;

  // Create a new FileReader() object
  let reader = new FileReader();

  // Setup the callback event to run when the file is read
  reader.onload = logFile;

  // Read the file
  reader.readAsDataURL(file.files[0]);
  console.log("read");
}

// Listen for submit events
form.addEventListener("submit", handleSubmit);
