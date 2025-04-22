FROM busybox:latest
COPY README.md /
# Don't need root for this.
USER nobody
# This uses C-style strings in the CVEs section of the readme, pipe via printf to expand them.
CMD sed -n '/^## CVEs/,/^##/p' /README.md | xargs -n1 printf > /dev/termination-log && false
