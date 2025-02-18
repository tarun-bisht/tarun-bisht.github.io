const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');
const theme = localStorage.getItem("theme");
handle_theme();
function handle_theme()
{
    if((theme === null && prefersDarkScheme.matches) || theme === "dark")
    {
        enable_dark_mode()
    }
    else
    {
        enable_light_mode();
    }
}
function enable_dark_mode()
{
    document.documentElement.setAttribute("data-theme","dark");
}
function enable_light_mode()
{
    document.documentElement.setAttribute("data-theme","light");
}