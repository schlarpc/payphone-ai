# payphone-ai

## Development

### Loading the environment

```shell
$ direnv allow
```

```shell
$ payphone-ai
```

```shell
$ direnv reload
```

### Maintaining Python dependencies

```shell
$ poetry add some-package
```

```shell
$ poetry update
```

```shell
$ poetryup
```

### Testing and linting

```shell
$ pre-commit run --all
```

### Using the Nix build system

```shell
$ nix run .
```

```shell
$ nix run '.#containerImage' | docker load
```

```shell
$ nix flake check
```

```shell
$ nix flake update
```

## Updating the base template

```shell
$ cruft update --checkout template
```
