// The page asks /api/... and Next passes it on to the Python server (server.py in the folder above).
const server = process.env.GAME_SERVER ?? "http://127.0.0.1:8000";

/** @type {import('next').NextConfig} */
const config = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${server}/api/:path*` }];
  },
};

export default config;
