// Get the form and file field
let form = document.querySelector("#upload");
let cta = document.querySelector("#cta");
let file = document.querySelector("#file");
let app = document.querySelector("#app");
let load = document.querySelector("#loading");
let get_started = document.querySelector("#next");
let dr = document.querySelector("#doctor");
let pt = document.querySelector("#patient");
let greet = document.querySelector("#greeting");

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
    status: localStorage.getItem("status"),
    image_name: ["ISIC_0251455"],
    patient_id: ["IP_3579794"],
    sex: [localStorage.getItem("sex")],
    age_approx: [localStorage.getItem("age_approx")],
    anatom_site_general_challenge: localStorage.getItem(
      "anatom_site_general_challenge"
    ),
    width: [6000],
    height: [4000],
  };
  formData.append("params", JSON.stringify(params));
  formData.append("photo", str);

  req.onreadystatechange = function () {
    if (req.readyState == XMLHttpRequest.DONE) {
      load.classList.remove("shown");
      load.classList.add("hidden");
      resp = JSON.parse(req.responseText);
      renderResult(resp.prob, resp.diag, resp.recs);
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

function renderFirst() {
  let f = document.querySelector("#first-page");
  f.hidden = true;
  let s = document.querySelector("#second-page");
  s.hidden = false;
}

function doctor() {
  let f = document.querySelector("#second-page");
  f.hidden = true;
  let s = document.querySelector("#third-page");
  s.hidden = false;
  greet.innerHTML = "Tell us more about your patient!";
  localStorage.setItem("status", "doctor");
}

function patient() {
  let f = document.querySelector("#second-page");
  f.hidden = true;
  let s = document.querySelector("#third-page");
  s.hidden = false;
  greet.innerHTML = "Tell us more about yourself!";
  localStorage.setItem("status", "patient");
}

function logData(event) {
  event.preventDefault();
  localStorage.setItem("sex", gender.value);
  localStorage.setItem("age_approx", age.value);
  localStorage.setItem("anatom_site_general_challenge", select_where.value);
  let f = document.querySelector("#third-page");
  f.hidden = true;
  let elem = document.getElementById("main-container");
  elem.style.width = "120rem";
  let ele = document.getElementById("final-page");
  ele.classList.remove("hidden");
}

function renderResult(prob, diagnosis, recs) {
  let diag = document.getElementById("diagnosis");
  let rec = document.getElementById("recs");
  let pred = document.getElementById("prediction");
  var options = {
    useEasing: true,
    useGrouping: true,
    separator: ",",
    decimal: ".",
    prefix: "",
    suffix: "%",
  };
  var demo = new CountUp("prediction", 0, prob, 2, 3, options);
  pred.hidden = false;
  demo.start();
  diag.innerHTML = diagnosis;
  diag.hidden = false;
  rec.innerHTML = recs;
  rec.hidden = false;
}

// Listen for submit events
get_started.addEventListener("click", renderFirst);
dr.addEventListener("click", doctor);
pt.addEventListener("click", patient);
cta.addEventListener("submit", logData);
form.addEventListener("submit", handleSubmit);
