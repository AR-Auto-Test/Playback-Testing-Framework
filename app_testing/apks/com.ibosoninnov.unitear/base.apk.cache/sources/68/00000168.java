package b.b.g;

import android.content.Context;
import android.content.ContextWrapper;
import android.content.res.AssetManager;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.view.LayoutInflater;

/* compiled from: ContextThemeWrapper.java */
/* loaded from: classes.dex */
public class c extends ContextWrapper {

    /* renamed from: a  reason: collision with root package name */
    public int f636a;

    /* renamed from: b  reason: collision with root package name */
    public Resources.Theme f637b;

    /* renamed from: c  reason: collision with root package name */
    public LayoutInflater f638c;

    /* renamed from: d  reason: collision with root package name */
    public Configuration f639d;

    /* renamed from: e  reason: collision with root package name */
    public Resources f640e;

    public c() {
        super(null);
    }

    public void a(Configuration configuration) {
        if (this.f640e == null) {
            if (this.f639d == null) {
                this.f639d = new Configuration(configuration);
                return;
            }
            throw new IllegalStateException("Override configuration has already been set");
        }
        throw new IllegalStateException("getResources() or getAssets() has already been called");
    }

    @Override // android.content.ContextWrapper
    public void attachBaseContext(Context context) {
        super.attachBaseContext(context);
    }

    public final void b() {
        if (this.f637b == null) {
            this.f637b = getResources().newTheme();
            Resources.Theme theme = getBaseContext().getTheme();
            if (theme != null) {
                this.f637b.setTo(theme);
            }
        }
        this.f637b.applyStyle(this.f636a, true);
    }

    @Override // android.content.ContextWrapper, android.content.Context
    public AssetManager getAssets() {
        return getResources().getAssets();
    }

    @Override // android.content.ContextWrapper, android.content.Context
    public Resources getResources() {
        if (this.f640e == null) {
            Configuration configuration = this.f639d;
            if (configuration == null) {
                this.f640e = super.getResources();
            } else {
                this.f640e = createConfigurationContext(configuration).getResources();
            }
        }
        return this.f640e;
    }

    @Override // android.content.ContextWrapper, android.content.Context
    public Object getSystemService(String str) {
        if ("layout_inflater".equals(str)) {
            if (this.f638c == null) {
                this.f638c = LayoutInflater.from(getBaseContext()).cloneInContext(this);
            }
            return this.f638c;
        }
        return getBaseContext().getSystemService(str);
    }

    @Override // android.content.ContextWrapper, android.content.Context
    public Resources.Theme getTheme() {
        Resources.Theme theme = this.f637b;
        if (theme != null) {
            return theme;
        }
        if (this.f636a == 0) {
            this.f636a = 2131886496;
        }
        b();
        return this.f637b;
    }

    @Override // android.content.ContextWrapper, android.content.Context
    public void setTheme(int i) {
        if (this.f636a != i) {
            this.f636a = i;
            b();
        }
    }

    public c(Context context, int i) {
        super(context);
        this.f636a = i;
    }

    public c(Context context, Resources.Theme theme) {
        super(context);
        this.f637b = theme;
    }
}