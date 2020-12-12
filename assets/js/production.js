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

const navMenu=function(animationName="")
{
    const menu_btn=document.querySelector(".menu-btn");
    const menu=document.querySelector(".menu");
    const menu_links=document.querySelectorAll(".menu li");
    menu_btn.addEventListener("click",function()
    {
        menu.classList.toggle("menu-active");
        if (document.body.style.overflow === "hidden")
        {
            document.body.style.overflow = "auto";
        } 
        else 
        {
            document.body.style.overflow = "hidden";
        }
        const menu_btn_child=menu_btn.children;
        menu_links.forEach(function(link,index){
            if(link.style.animation)
            {
                link.style.animation="";
                menu_btn_child[0].style="";
                menu_btn_child[1].style="";
                menu_btn_child[2].style="";
            }
            else
            {
                link.style.animation=animationName+" 0.5s ease forwards "+(index / 7 + 0.3).toString()+"s";
                menu_btn_child[0].style="transform:rotate(-45deg) translate(-6px,6px);";
                menu_btn_child[1].style="opacity:0;";
                menu_btn_child[2].style="transform:rotate(45deg) translate(-6px,-6px)";
            }
        });
    });
}
const stickyNav=function(navMenu,top)
{
    window.onscroll=function()
    {
        navMenu.classList.toggle("nav-fixed",window.scrollY > top);
    }
}
const activeLinkOfPage=function(menu)
{
    const curr_location=window.location.href;
    const links=menu.querySelectorAll("a");
    links.forEach(function(link)
    {
        if(link.href === curr_location)
        {
            link.className="nav-active";
        }
    });
}
const observeContainers=function(containers,options,onIntersect=function(element){},furtherObserve=false)
{
    const observer=new IntersectionObserver(function(entries,observer)
    {
        entries.forEach(function(elem)
        {
            if(elem.isIntersecting)
            {
                onIntersect(elem.target)
                if(!furtherObserve)
                {
                    observer.unobserve(elem.target);
                }
            }
        });
    },options);
    containers.forEach(function(container)
    {
        observer.observe(container);
    });
}
const filterCategory=function(objects,category)
{
    const objs=Array.from(objects);
    hideAll(objs);
    if(category==="all")
    {
        showAll(objs);
    }
    else
    {
        const filtered = objs.filter(function(object)
        {
            return object.getAttribute("category")==category;
        });
        showAll(filtered);
    }
}
const hideAll=function(objects)
{
    objects.forEach(function(object)
    {
        object.style.display="none";
    });
}
const showAll=function(objects)
{
    objects.forEach(function(object)
    {
        object.style.display="block";
    });
}
const get_params=function() 
{
    let vars = {};
    let parts = window.location.href.replace(/[?&]+([^=&]+)=([^&]*)/gi, function(m,key,value) 
    {
        vars[key] = value;
    });
    return vars;
}
function html_transition()
{
    document.documentElement.classList.add('transition');
    window.setTimeout(function()
    {
        document.documentElement.classList.remove('transition');
    },1000);
}
function dark_mode()
{
    document.documentElement.setAttribute("data-theme","dark");
    localStorage.setItem("dark",true);
}
function light_mode()
{
    document.documentElement.setAttribute("data-theme","light");
    localStorage.setItem("dark",null);
}
function isdarkMode()
{
    if(localStorage.getItem("dark")==="true")
    {
        dark_mode();
        document.getElementById("switch").checked=true;
    }
    else
    {
        light_mode();
        document.getElementById("switch").checked=false;
    }
}