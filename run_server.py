#!/usr/bin/env python3
"""Run the tongue API server."""

import uvicorn


def main():
    print("Starting Tongue API server...")
    print("API documentation available at: http://localhost:8000/docs")
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()
