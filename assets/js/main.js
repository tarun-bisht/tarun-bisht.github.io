const imgOptions={
    threshold:0,
    rootMargin:"0px 0px 300px 0px"
};
isdarkMode();
window.onload=function()
{
    nav_menu();
    animate_containers();
    filter_projects();
    LazyLoad();
    dark_mode_switch();
}
function nav_menu()
{
    navMenu("fade-right");
    const navbar=document.querySelector("nav");
    stickyNav(navbar,300);
    activeLinkOfPage(document.querySelector(".menu"));
}
function animate_containers()
{
    const animation_containers=document.querySelectorAll('[animation]');
    observeContainers(animation_containers,{threshold:0},onIntersect=animateContainer,furtherObserve=false);
}

function filter_projects()
{
    const filterLinks=document.querySelectorAll('[data-category]');
    filterLinks.forEach(function(link)
    {
        link.onclick=function(evt)
        {
            const projects=document.querySelectorAll('[category]');
            category=evt.target.getAttribute("data-category");
            filterCategory(projects,category);
        }
    });
}
function animateContainer(element)
{
    const anim=element.getAttribute("animation");
    const anim_time=element.getAttribute("animation-time");
    if(anim)
    {
        element.style.animation=`${anim} ${anim_time} ease-in-out`;
    }
}
function LazyLoad()
{
    const tags=document.querySelectorAll('[data-src]');
    if(tags)
    {
        observeContainers(tags,{threshold:0,rootMargin:"0px 0px 300px 0px"},onIntersect=preload,furtherObserve=false);
    }
}
function preload(tag)
{
    const src=tag.getAttribute("data-src");
    if(src)
    {
        tag.src=src;
    }
}
function dark_mode_switch()
{
    let switch_box=document.getElementById("switch");
    switch_box.addEventListener("change",function()
    {
        html_transition();
        if(this.checked)
        {
            dark_mode();
        }
        else
        {
            light_mode();
        }
    });
}
function fontawesome_fallback(cssFileToCheck)
{
    var styleSheets = document.styleSheets;
    for (var i = 0, max = styleSheets.length; i < max; i++) 
    {
        if (styleSheets[i].href == cssFileToCheck) 
        {
            return;
        }
    }
    // because no matching stylesheets were found, we will add a new HTML link element to the HEAD section of the page.
    var link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "assets/css/font-awesome.min.css";
    document.getElementsByTagName("head")[0].appendChild(link);
}
(function(d) {
    var wf = d.createElement('script'), s = d.scripts[0];
    wf.src = 'https://ajax.googleapis.com/ajax/libs/webfont/1.6.26/webfont.js';
    wf.async = true;
    s.parentNode.insertBefore(wf, s);
})(document);
WebFontConfig = {
    google: { families: ['Poppins'] }
};
fontawesome_fallback("https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.13.0/css/all.min.css");