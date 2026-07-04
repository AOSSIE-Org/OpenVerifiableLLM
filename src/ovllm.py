#!/usr/bin/env python3
import argparse
import sys

from publish import build_ollama, prepare_publish_dir, publish_huggingface, sign_model_dir
from verifier import print_report, verify_model_reference


def _verify(args) -> int:
    try:
        results = verify_model_reference(
            args.model_ref,
            cache_dir=args.cache_dir,
            allow_unsigned=args.allow_unsigned,
            skip_replay=args.skip_replay,
        )
    except Exception as exc:
        print(f"[FAIL] verifier_error - {exc}")
        print("\nVERDICT: RED")
        return 1
    return 0 if print_report(results) else 1


def _prepare_publish(args) -> int:
    out = prepare_publish_dir(weights=args.weights, output_dir=args.out, name=args.name)
    print(f"Prepared publish directory: {out}")
    return 0


def _sign(args) -> int:
    return sign_model_dir(
        args.model_dir,
        signature=args.signature,
        identity=args.identity,
        identity_provider=args.identity_provider,
        use_ambient_credentials=args.use_ambient_credentials,
        dry_run=args.dry_run,
    )


def _publish_hf(args) -> int:
    return publish_huggingface(args.repo_id, args.model_dir, dry_run=args.dry_run)


def _ollama(args) -> int:
    return build_ollama(args.name, args.model_dir, dry_run=args.dry_run)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ovllm", description="OpenVerifiableLLM CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    verify = sub.add_parser("verify", help="verify a local path or Hugging Face model reference")
    verify.add_argument("model_ref")
    verify.add_argument("--cache-dir", default=None)
    verify.add_argument("--allow-unsigned", action="store_true",
                        help="treat a missing Sigstore bundle as SKIP instead of FAIL")
    verify.add_argument("--skip-replay", action="store_true")
    verify.set_defaults(func=_verify)

    prep = sub.add_parser("prepare-publish", help="build a publishable model directory")
    prep.add_argument("--weights", required=True)
    prep.add_argument("--out", required=True)
    prep.add_argument("--name", default="openverifiable-small")
    prep.set_defaults(func=_prepare_publish)

    sign = sub.add_parser("sign", help="sign a model directory with sigstore/model-transparency")
    sign.add_argument("model_dir")
    sign.add_argument("--signature", default="model.sig")
    sign.add_argument("--identity", default=None,
                      help="expected Sigstore signer identity to store in ovllm_manifest.json")
    sign.add_argument("--identity-provider", dest="identity_provider", default=None,
                      help="expected Sigstore identity provider URL to store in ovllm_manifest.json")
    sign.add_argument("--use-ambient-credentials", action="store_true",
                      help="use ambient OIDC credentials, e.g. GitHub Actions id-token")
    sign.add_argument("--dry-run", action="store_true")
    sign.set_defaults(func=_sign)

    hf = sub.add_parser("publish-hf", help="upload a prepared model directory to Hugging Face")
    hf.add_argument("repo_id")
    hf.add_argument("model_dir")
    hf.add_argument("--dry-run", action="store_true")
    hf.set_defaults(func=_publish_hf)

    ollama = sub.add_parser("ollama-build", help="run ollama create from the generated Modelfile")
    ollama.add_argument("name")
    ollama.add_argument("model_dir")
    ollama.add_argument("--dry-run", action="store_true")
    ollama.set_defaults(func=_ollama)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
