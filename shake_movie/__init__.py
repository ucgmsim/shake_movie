"""Shake movies: public-facing videos of an SW4 simulation.

The movie is built entirely from files a workflow run already has: SW4's
surface images (`topo`, `surf_mag`, `surf_hmax`), the SRF SW4 ran, the
realisation and, for the final map, `im-calc`'s broadband intensity measures.
Nothing is configured per event, so the realtime workflow can
render and upload a movie for every earthquake it simulates.
"""
