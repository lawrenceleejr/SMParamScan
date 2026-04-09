FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends gfortran wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN wget -q -O hdecay.f https://www.fuw.edu.pl/~kalino/fortran/hdecay.f && \
    sed -i 's/COMPLEX FUNCTION F0\*16/COMPLEX*16 FUNCTION F0/' hdecay.f && \
    sed -i 's/COMPLEX FUNCTION LI2\*16/COMPLEX*16 FUNCTION LI2/' hdecay.f && \
    sed -i 's/COMPLEX FUNCTION CLI2\*16/COMPLEX*16 FUNCTION CLI2/' hdecay.f && \
    gfortran -O2 -std=legacy -o hdecay hdecay.f

RUN mkdir /app/work
WORKDIR /app/work

ENTRYPOINT ["/app/hdecay"]
