const util = require('util');
const exec = util.promisify(require('child_process').exec);
const fs = require("fs").promises


async function runModel(lat, long) {
    try {
        const { stdout, stderr } = await exec(`venv/bin/python3 main.py ${lat} ${long}`, {maxBuffer: 1024 * 5000});
  
        const outputArr = stdout.split("\n");
        const newfile = outputArr[outputArr.length - 2]
    
        // Read in the file
        const jsonData = await fs.readFile(newfile)
        console.log(JSON.parse(jsonData));
    
        return JSON.parse(jsonData);
    } catch (e) {
        console.log("error getting data", e);
        return {};
    }
}

runModel(40.02882154070936, -121.14185813242528);