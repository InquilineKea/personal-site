const localtunnel = require('localtunnel');
const fs = require('fs');

(async () => {
  const tunnel = await localtunnel({ port: 5000 });

  const message = `
================================================================================
🚀 Your Flask app is now publicly accessible!
🌐 Public URL: ${tunnel.url}
📱 Share this URL to access your site from anywhere!
================================================================================
`;

  console.log(message);
  fs.writeFileSync('/tmp/tunnel_url.txt', tunnel.url);
  fs.writeFileSync('/tmp/tunnel_output.txt', message);

  tunnel.on('close', () => {
    console.log('Tunnel closed');
  });

  // Keep the process running
  process.on('SIGINT', () => {
    tunnel.close();
    process.exit();
  });
})();
