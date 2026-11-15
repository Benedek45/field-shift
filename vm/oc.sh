#!/bin/sh
# Run the pinned OpenCode CLI in a container against one run directory:
#   vm/oc.sh <run-dir> [opencode args...]
# <run-dir> must hold cfg/, data/ and work/. Only variables named in
# FIELDSHIFT_PASS_ENV (space separated) are passed through, by name (`-e NAME`),
# so key values never appear on a command line. fieldshift.agent.providers sets both.
set -eu
run_dir=$1
shift
env_args=""
for name in ${FIELDSHIFT_PASS_ENV:-}; do
    case "$name" in
        *[!A-Za-z0-9_]* | [0-9]*)
            echo "FIELDSHIFT_PASS_ENV must list variable names only" >&2
            exit 2
            ;;
    esac
    env_args="$env_args -e $name"
done
name_args=""
if [ -n "${FIELDSHIFT_CONTAINER_NAME:-}" ]; then
    name_args="--name $FIELDSHIFT_CONTAINER_NAME"
fi
case "${FIELDSHIFT_ANTHROPIC:-0}${FIELDSHIFT_OPENAI:-0}" in
    *1*)
        echo "Subscription-login adapters are not part of this repository; use an API key" \
            "(e.g. anthropic/* with ANTHROPIC_API_KEY, openrouter/* with OPENROUTER_API_KEY)." >&2
        exit 2
        ;;
esac
# shellcheck disable=SC2086
exec docker --context "${FIELDSHIFT_DOCKER_CONTEXT:-colima-rosetta}" run --rm -i $name_args \
    --add-host host.docker.internal:host-gateway \
    --mount "type=bind,src=$run_dir/cfg,dst=/cfg" \
    --mount "type=bind,src=$run_dir/data,dst=/data" \
    --mount "type=bind,src=$run_dir/work,dst=/work" \
    $env_args \
    --entrypoint opencode "${FIELDSHIFT_IMAGE:-fieldshift-agent:oc2.3.0-arm64}" "$@"
