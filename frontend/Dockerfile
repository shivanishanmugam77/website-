# syntax=docker/dockerfile:1
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
# Reproducible when a lockfile is committed (it should be); falls back to install otherwise.
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

# ---- development image (used by docker-compose): source is bind-mounted
FROM node:20-alpine AS dev
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
COPY --from=deps --chown=node:node /app/node_modules ./node_modules
# Pre-create the cache dir so the named volume mounted here inherits node ownership.
RUN mkdir -p /app/.next && chown node:node /app /app/.next
COPY --chown=node:node . .
USER node
EXPOSE 3000
CMD ["npm", "run", "dev"]

FROM node:20-alpine AS build
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# ---- production image (default target): Next.js standalone output, unprivileged
FROM node:20-alpine AS prod
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 PORT=3000 HOSTNAME=0.0.0.0
COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static
COPY --from=build --chown=node:node /app/public ./public
USER node
EXPOSE 3000
CMD ["node", "server.js"]
