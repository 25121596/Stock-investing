const fs = require('fs');
const path = require('path');
const {pathToFileURL, fileURLToPath} = require('url');
const {chromium} = require('C:/Users/zzczz/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
  const folder=path.resolve(process.argv[2] || 'results/exploration-v2');
  const reportName=process.argv[3] || 'exploration-report.html';
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const rows=[];
  try {
    for(const [name,width,height] of [['desktop',1440,1000],['mobile',390,844]]) {
      const page=await browser.newPage({viewport:{width,height}});
      await page.goto(pathToFileURL(path.join(folder,reportName)).href);
      const r=await page.evaluate(()=>({title:document.title,width:innerWidth,
        scrollWidth:document.documentElement.scrollWidth,links:[...document.querySelectorAll('a[href]')].map(a=>a.href)}));
      const missing=r.links.filter(u=>u.startsWith('file:')&&!fs.existsSync(fileURLToPath(u)));
      rows.push({name,width,scrollWidth:r.scrollWidth,links:r.links.length,missing,passed:r.scrollWidth<=width&&!missing.length});
      await page.screenshot({path:path.join(folder,`preview-${name}.png`)});
      await page.close();
    }
    fs.writeFileSync(path.join(folder,'layout-checks.json'),JSON.stringify(rows,null,2));
    console.log(JSON.stringify(rows));
    if(rows.some(r=>!r.passed))process.exitCode=1;
  } finally {await browser.close();}
})();
