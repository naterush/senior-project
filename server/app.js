var express = require('express');
const util = require('util');
var morgan = require('morgan');
const fs = require(`fs`).promises;
const exec = util.promisify(require('child_process').exec);
var app = express();

app.use(morgan("dev"));

app.engine('html', require('ejs').__express);
app.set('view engine', 'html');
app.use(express.static(__dirname + '/public'));
app.set('view engine', 'ejs');

app.use(express.urlencoded());

async function runModel(lat, long) {
  try {
    console.log("EXECING");
    const { stdout, stderr } = await exec(`venv/bin/python3 main.py ${lat} ${long}`, {maxBuffer: 1024 * 5000});
    console.log("DONE EXECING");

    const outputArr = stdout.split("\n");
    const newfile = outputArr[outputArr.length - 2]

    console.log("GOT NEW FILE", newfile)
  
    // Read in the file
    const jsonData = await fs.readFile(newfile)
    console.log("GOT NEW FILE")

    return JSON.parse(jsonData);
  } catch (e) {
    console.log("error getting data", e);
    return {};
  }
}

app.get('/', async function (req, res) {
  res.render('index.html');
});


app.get('/map', async function (req, res) {
  res.render('map.html');
});


app.get('/getRegion', async function (req, res) {
  // We set a timeout of 15 minutes...
  // because the model might take a while
  req.setTimeout(15 * 60 * 1000);

  const {latitude, longitude} = req.query;

  console.log(`Getting region ${latitude}, ${longitude}. This may take a while`);

  const modelJSONResult = await runModel(latitude, longitude);

  console.log(`Returning result ${latitude}, ${longitude} with : ${modelJSONResult.length} results`);

  res.json(modelJSONResult);
});

app.listen(3000, () => {
  console.log("Server is running on port 3000");
});