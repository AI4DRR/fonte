# Fonte Pages

This repository contains the source files for the public Fonte website:

https://ai4drr.github.io/fonte/

## How It Works

Changes committed to the `main` branch are automatically deployed to the `pages/` directory of the main Fonte repository.

Deployment flow:

    AI4DRR/fonte-pages
            ↓
      GitHub Actions
            ↓
    AI4DRR/fonte/pages
            ↓
       GitHub Pages
            ↓
    https://ai4drr.github.io/fonte/

There is no need to manually copy website files to the main Fonte repository.

## Editing the Website

Website pages and assets can be created and edited directly in this repository.

For example, you can add:

    index.html
    about.html
    contact.html
    css/
    js/
    images/

After the changes are committed to `main`, they will be automatically deployed to the public website.

## Adding a New Page

For example, create:

    about.html

Commit the file to `main`.

Once deployment completes, the page will be available at:

    https://ai4drr.github.io/fonte/about.html

The same process applies to HTML pages, stylesheets, JavaScript, images, and other static website assets.

## Example Structure

The website structure is flexible. The example below shows one possible way to organize the files:

    fonte-pages/
    ├── .github/
    │   └── workflows/
    │       └── deploy.yml    # Automatic deployment - do not modify
    ├── index.html
    ├── about.html
    ├── css/
    │   └── style.css
    ├── js/
    │   └── main.js
    └── images/
        └── logo.png

This is only an example. Website files and directories can be organized as needed.

## Important

Do **not** modify or delete:

    .github/workflows/deploy.yml

This workflow handles the automatic deployment from this repository to the main Fonte repository.

Also, do not commit passwords, API keys, access tokens, or other secrets to this repository.

## Deleting or Renaming Files

The deployment mirrors the website files in this repository to the published `pages/` directory.

If a website file is deleted or renamed here, the corresponding published file will also be deleted or renamed during the next deployment.

## Deployment

Deployment is automatic whenever website files are committed to the `main` branch.

To check the deployment status, open the **Actions** tab in this repository.

Public website:

https://ai4drr.github.io/fonte/
