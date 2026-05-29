import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sklearn

    return (sklearn,)


@app.cell
def _(sklearn):
    # Version
    sklearn.show_versions()
    return


@app.cell
def _(sklearn):
    # Load config
    sklearn.get_config()
    return


if __name__ == "__main__":
    app.run()
