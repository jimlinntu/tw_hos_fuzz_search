#!/bin/bash
gunicorn -w "${WORKER:-10}" -b '0.0.0.0:80' tw_hos_fuzz_search.server:app
