package b.b.g;

import android.app.Activity;
import android.content.Context;
import android.content.ContextWrapper;
import android.content.res.ColorStateList;
import android.content.res.TypedArray;
import android.content.res.XmlResourceParser;
import android.graphics.PorterDuff;
import android.os.Build;
import android.util.AttributeSet;
import android.util.Log;
import android.util.Xml;
import android.view.InflateException;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.SubMenu;
import android.view.View;
import b.b.g.i.i;
import b.b.g.i.j;
import b.b.h.e0;
import b.b.h.y0;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import org.xmlpull.v1.XmlPullParser;
import org.xmlpull.v1.XmlPullParserException;

/* compiled from: SupportMenuInflater.java */
/* loaded from: classes.dex */
public class f extends MenuInflater {

    /* renamed from: a  reason: collision with root package name */
    public static final Class<?>[] f652a;

    /* renamed from: b  reason: collision with root package name */
    public static final Class<?>[] f653b;

    /* renamed from: c  reason: collision with root package name */
    public final Object[] f654c;

    /* renamed from: d  reason: collision with root package name */
    public final Object[] f655d;

    /* renamed from: e  reason: collision with root package name */
    public Context f656e;

    /* renamed from: f  reason: collision with root package name */
    public Object f657f;

    /* compiled from: SupportMenuInflater.java */
    /* loaded from: classes.dex */
    public static class a implements MenuItem.OnMenuItemClickListener {

        /* renamed from: a  reason: collision with root package name */
        public static final Class<?>[] f658a = {MenuItem.class};

        /* renamed from: b  reason: collision with root package name */
        public Object f659b;

        /* renamed from: c  reason: collision with root package name */
        public Method f660c;

        public a(Object obj, String str) {
            this.f659b = obj;
            Class<?> cls = obj.getClass();
            try {
                this.f660c = cls.getMethod(str, f658a);
            } catch (Exception e2) {
                StringBuilder B = c.b.a.a.a.B("Couldn't resolve menu item onClick handler ", str, " in class ");
                B.append(cls.getName());
                InflateException inflateException = new InflateException(B.toString());
                inflateException.initCause(e2);
                throw inflateException;
            }
        }

        @Override // android.view.MenuItem.OnMenuItemClickListener
        public boolean onMenuItemClick(MenuItem menuItem) {
            try {
                if (this.f660c.getReturnType() == Boolean.TYPE) {
                    return ((Boolean) this.f660c.invoke(this.f659b, menuItem)).booleanValue();
                }
                this.f660c.invoke(this.f659b, menuItem);
                return true;
            } catch (Exception e2) {
                throw new RuntimeException(e2);
            }
        }
    }

    /* compiled from: SupportMenuInflater.java */
    /* loaded from: classes.dex */
    public class b {
        public b.j.j.b A;
        public CharSequence B;
        public CharSequence C;

        /* renamed from: a  reason: collision with root package name */
        public Menu f661a;

        /* renamed from: h  reason: collision with root package name */
        public boolean f668h;
        public int i;
        public int j;
        public CharSequence k;
        public CharSequence l;
        public int m;
        public char n;
        public int o;
        public char p;
        public int q;
        public int r;
        public boolean s;
        public boolean t;
        public boolean u;
        public int v;
        public int w;
        public String x;
        public String y;
        public String z;
        public ColorStateList D = null;
        public PorterDuff.Mode E = null;

        /* renamed from: b  reason: collision with root package name */
        public int f662b = 0;

        /* renamed from: c  reason: collision with root package name */
        public int f663c = 0;

        /* renamed from: d  reason: collision with root package name */
        public int f664d = 0;

        /* renamed from: e  reason: collision with root package name */
        public int f665e = 0;

        /* renamed from: f  reason: collision with root package name */
        public boolean f666f = true;

        /* renamed from: g  reason: collision with root package name */
        public boolean f667g = true;

        public b(Menu menu) {
            this.f661a = menu;
        }

        public SubMenu a() {
            this.f668h = true;
            SubMenu addSubMenu = this.f661a.addSubMenu(this.f662b, this.i, this.j, this.k);
            c(addSubMenu.getItem());
            return addSubMenu;
        }

        public final <T> T b(String str, Class<?>[] clsArr, Object[] objArr) {
            try {
                Constructor<?> constructor = Class.forName(str, false, f.this.f656e.getClassLoader()).getConstructor(clsArr);
                constructor.setAccessible(true);
                return (T) constructor.newInstance(objArr);
            } catch (Exception e2) {
                Log.w("SupportMenuInflater", "Cannot instantiate class: " + str, e2);
                return null;
            }
        }

        public final void c(MenuItem menuItem) {
            boolean z = false;
            menuItem.setChecked(this.s).setVisible(this.t).setEnabled(this.u).setCheckable(this.r >= 1).setTitleCondensed(this.l).setIcon(this.m);
            int i = this.v;
            if (i >= 0) {
                menuItem.setShowAsAction(i);
            }
            if (this.z != null) {
                if (!f.this.f656e.isRestricted()) {
                    f fVar = f.this;
                    if (fVar.f657f == null) {
                        fVar.f657f = fVar.a(fVar.f656e);
                    }
                    menuItem.setOnMenuItemClickListener(new a(fVar.f657f, this.z));
                } else {
                    throw new IllegalStateException("The android:onClick attribute cannot be used within a restricted context");
                }
            }
            if (this.r >= 2) {
                if (menuItem instanceof i) {
                    ((i) menuItem).k(true);
                } else if (menuItem instanceof j) {
                    j jVar = (j) menuItem;
                    try {
                        if (jVar.f740e == null) {
                            jVar.f740e = jVar.f739d.getClass().getDeclaredMethod("setExclusiveCheckable", Boolean.TYPE);
                        }
                        jVar.f740e.invoke(jVar.f739d, Boolean.TRUE);
                    } catch (Exception e2) {
                        Log.w("MenuItemWrapper", "Error while calling setExclusiveCheckable", e2);
                    }
                }
            }
            String str = this.x;
            if (str != null) {
                menuItem.setActionView((View) b(str, f.f652a, f.this.f654c));
                z = true;
            }
            int i2 = this.w;
            if (i2 > 0) {
                if (!z) {
                    menuItem.setActionView(i2);
                } else {
                    Log.w("SupportMenuInflater", "Ignoring attribute 'itemActionViewLayout'. Action view already specified.");
                }
            }
            b.j.j.b bVar = this.A;
            if (bVar != null) {
                if (menuItem instanceof b.j.e.a.b) {
                    ((b.j.e.a.b) menuItem).a(bVar);
                } else {
                    Log.w("MenuItemCompat", "setActionProvider: item does not implement SupportMenuItem; ignoring");
                }
            }
            CharSequence charSequence = this.B;
            boolean z2 = menuItem instanceof b.j.e.a.b;
            if (z2) {
                ((b.j.e.a.b) menuItem).setContentDescription(charSequence);
            } else if (Build.VERSION.SDK_INT >= 26) {
                menuItem.setContentDescription(charSequence);
            }
            CharSequence charSequence2 = this.C;
            if (z2) {
                ((b.j.e.a.b) menuItem).setTooltipText(charSequence2);
            } else if (Build.VERSION.SDK_INT >= 26) {
                menuItem.setTooltipText(charSequence2);
            }
            char c2 = this.n;
            int i3 = this.o;
            if (z2) {
                ((b.j.e.a.b) menuItem).setAlphabeticShortcut(c2, i3);
            } else if (Build.VERSION.SDK_INT >= 26) {
                menuItem.setAlphabeticShortcut(c2, i3);
            }
            char c3 = this.p;
            int i4 = this.q;
            if (z2) {
                ((b.j.e.a.b) menuItem).setNumericShortcut(c3, i4);
            } else if (Build.VERSION.SDK_INT >= 26) {
                menuItem.setNumericShortcut(c3, i4);
            }
            PorterDuff.Mode mode = this.E;
            if (mode != null) {
                if (z2) {
                    ((b.j.e.a.b) menuItem).setIconTintMode(mode);
                } else if (Build.VERSION.SDK_INT >= 26) {
                    menuItem.setIconTintMode(mode);
                }
            }
            ColorStateList colorStateList = this.D;
            if (colorStateList != null) {
                if (z2) {
                    ((b.j.e.a.b) menuItem).setIconTintList(colorStateList);
                } else if (Build.VERSION.SDK_INT >= 26) {
                    menuItem.setIconTintList(colorStateList);
                }
            }
        }
    }

    static {
        Class<?>[] clsArr = {Context.class};
        f652a = clsArr;
        f653b = clsArr;
    }

    public f(Context context) {
        super(context);
        this.f656e = context;
        Object[] objArr = {context};
        this.f654c = objArr;
        this.f655d = objArr;
    }

    public final Object a(Object obj) {
        return (!(obj instanceof Activity) && (obj instanceof ContextWrapper)) ? a(((ContextWrapper) obj).getBaseContext()) : obj;
    }

    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:95:0x022f */
    public final void b(XmlPullParser xmlPullParser, AttributeSet attributeSet, Menu menu) {
        ColorStateList colorStateList;
        b bVar = new b(menu);
        int eventType = xmlPullParser.getEventType();
        while (true) {
            if (eventType == 2) {
                String name = xmlPullParser.getName();
                if (name.equals("menu")) {
                    eventType = xmlPullParser.next();
                } else {
                    throw new RuntimeException(c.b.a.a.a.q("Expecting menu, got ", name));
                }
            } else {
                eventType = xmlPullParser.next();
                if (eventType == 1) {
                    break;
                }
            }
        }
        String str = null;
        boolean z = false;
        boolean z2 = false;
        while (!z) {
            if (eventType != 1) {
                z = z;
                z = z;
                if (eventType != 2) {
                    if (eventType == 3) {
                        String name2 = xmlPullParser.getName();
                        if (z2 && name2.equals(str)) {
                            str = null;
                            z2 = false;
                        } else if (name2.equals("group")) {
                            bVar.f662b = 0;
                            bVar.f663c = 0;
                            bVar.f664d = 0;
                            bVar.f665e = 0;
                            bVar.f666f = true;
                            bVar.f667g = true;
                            z = z;
                        } else if (name2.equals("item")) {
                            z = z;
                            if (!bVar.f668h) {
                                b.j.j.b bVar2 = bVar.A;
                                if (bVar2 != null && bVar2.a()) {
                                    bVar.a();
                                    z = z;
                                } else {
                                    bVar.f668h = true;
                                    bVar.c(bVar.f661a.add(bVar.f662b, bVar.i, bVar.j, bVar.k));
                                    z = z;
                                }
                            }
                        } else {
                            z = z;
                            if (name2.equals("menu")) {
                                z = true;
                            }
                        }
                    }
                } else if (!z2) {
                    String name3 = xmlPullParser.getName();
                    if (name3.equals("group")) {
                        TypedArray obtainStyledAttributes = f.this.f656e.obtainStyledAttributes(attributeSet, b.b.b.p);
                        bVar.f662b = obtainStyledAttributes.getResourceId(1, 0);
                        bVar.f663c = obtainStyledAttributes.getInt(3, 0);
                        bVar.f664d = obtainStyledAttributes.getInt(4, 0);
                        bVar.f665e = obtainStyledAttributes.getInt(5, 0);
                        bVar.f666f = obtainStyledAttributes.getBoolean(2, true);
                        bVar.f667g = obtainStyledAttributes.getBoolean(0, true);
                        obtainStyledAttributes.recycle();
                        z = z;
                    } else if (name3.equals("item")) {
                        y0 q = y0.q(f.this.f656e, attributeSet, b.b.b.q);
                        bVar.i = q.m(2, 0);
                        bVar.j = (q.j(5, bVar.f663c) & (-65536)) | (q.j(6, bVar.f664d) & 65535);
                        bVar.k = q.o(7);
                        bVar.l = q.o(8);
                        bVar.m = q.m(0, 0);
                        String n = q.n(9);
                        bVar.n = n == null ? (char) 0 : n.charAt(0);
                        bVar.o = q.j(16, 4096);
                        String n2 = q.n(10);
                        bVar.p = n2 == null ? (char) 0 : n2.charAt(0);
                        bVar.q = q.j(20, 4096);
                        if (q.p(11)) {
                            bVar.r = q.a(11, false) ? 1 : 0;
                        } else {
                            bVar.r = bVar.f665e;
                        }
                        bVar.s = q.a(3, false);
                        bVar.t = q.a(4, bVar.f666f);
                        bVar.u = q.a(1, bVar.f667g);
                        bVar.v = q.j(21, -1);
                        bVar.z = q.n(12);
                        bVar.w = q.m(13, 0);
                        bVar.x = q.n(15);
                        String n3 = q.n(14);
                        bVar.y = n3;
                        boolean z3 = n3 != null;
                        if (z3 && bVar.w == 0 && bVar.x == null) {
                            bVar.A = (b.j.j.b) bVar.b(n3, f653b, f.this.f655d);
                        } else {
                            if (z3) {
                                Log.w("SupportMenuInflater", "Ignoring attribute 'actionProviderClass'. Action view already specified.");
                            }
                            bVar.A = null;
                        }
                        bVar.B = q.o(17);
                        bVar.C = q.o(22);
                        if (q.p(19)) {
                            bVar.E = e0.c(q.j(19, -1), bVar.E);
                            colorStateList = null;
                        } else {
                            colorStateList = null;
                            bVar.E = null;
                        }
                        if (q.p(18)) {
                            bVar.D = q.c(18);
                        } else {
                            bVar.D = colorStateList;
                        }
                        q.f972b.recycle();
                        bVar.f668h = false;
                        z = z;
                    } else if (name3.equals("menu")) {
                        b(xmlPullParser, attributeSet, bVar.a());
                    } else {
                        str = name3;
                        z2 = true;
                    }
                }
                eventType = xmlPullParser.next();
                z = z;
                z2 = z2;
            } else {
                throw new RuntimeException("Unexpected end of document");
            }
        }
    }

    @Override // android.view.MenuInflater
    public void inflate(int i, Menu menu) {
        if (!(menu instanceof b.j.e.a.a)) {
            super.inflate(i, menu);
            return;
        }
        XmlResourceParser xmlResourceParser = null;
        try {
            try {
                try {
                    xmlResourceParser = this.f656e.getResources().getLayout(i);
                    b(xmlResourceParser, Xml.asAttributeSet(xmlResourceParser), menu);
                    xmlResourceParser.close();
                } catch (IOException e2) {
                    throw new InflateException("Error inflating menu XML", e2);
                }
            } catch (XmlPullParserException e3) {
                throw new InflateException("Error inflating menu XML", e3);
            }
        } catch (Throwable th) {
            if (xmlResourceParser != null) {
                xmlResourceParser.close();
            }
            throw th;
        }
    }
}