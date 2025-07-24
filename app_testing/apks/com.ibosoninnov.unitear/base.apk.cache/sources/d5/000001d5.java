package b.b.h;

import android.content.Context;
import android.content.res.ColorStateList;
import android.content.res.Resources;
import android.content.res.XmlResourceParser;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffColorFilter;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.LayerDrawable;
import android.util.AttributeSet;
import android.util.Log;
import android.util.TypedValue;
import android.util.Xml;
import b.b.h.j;
import com.ibosoninnov.unitear.R;
import java.lang.ref.WeakReference;
import java.util.Objects;
import java.util.WeakHashMap;
import org.xmlpull.v1.XmlPullParser;
import org.xmlpull.v1.XmlPullParserException;

/* compiled from: ResourceManagerInternal.java */
/* loaded from: classes.dex */
public final class n0 {

    /* renamed from: b  reason: collision with root package name */
    public static n0 f887b;

    /* renamed from: d  reason: collision with root package name */
    public WeakHashMap<Context, b.f.i<ColorStateList>> f889d;

    /* renamed from: e  reason: collision with root package name */
    public b.f.h<String, b> f890e;

    /* renamed from: f  reason: collision with root package name */
    public b.f.i<String> f891f;

    /* renamed from: g  reason: collision with root package name */
    public final WeakHashMap<Context, b.f.e<WeakReference<Drawable.ConstantState>>> f892g = new WeakHashMap<>(0);

    /* renamed from: h  reason: collision with root package name */
    public TypedValue f893h;
    public boolean i;
    public c j;

    /* renamed from: a  reason: collision with root package name */
    public static final PorterDuff.Mode f886a = PorterDuff.Mode.SRC_IN;

    /* renamed from: c  reason: collision with root package name */
    public static final a f888c = new a(6);

    /* compiled from: ResourceManagerInternal.java */
    /* loaded from: classes.dex */
    public static class a extends b.f.f<Integer, PorterDuffColorFilter> {
        public a(int i) {
            super(i);
        }
    }

    /* compiled from: ResourceManagerInternal.java */
    /* loaded from: classes.dex */
    public interface b {
        Drawable a(Context context, XmlPullParser xmlPullParser, AttributeSet attributeSet, Resources.Theme theme);
    }

    /* compiled from: ResourceManagerInternal.java */
    /* loaded from: classes.dex */
    public interface c {
    }

    public static synchronized n0 c() {
        n0 n0Var;
        synchronized (n0.class) {
            if (f887b == null) {
                f887b = new n0();
            }
            n0Var = f887b;
        }
        return n0Var;
    }

    public static synchronized PorterDuffColorFilter g(int i, PorterDuff.Mode mode) {
        PorterDuffColorFilter porterDuffColorFilter;
        synchronized (n0.class) {
            a aVar = f888c;
            Objects.requireNonNull(aVar);
            int i2 = (i + 31) * 31;
            porterDuffColorFilter = aVar.get(Integer.valueOf(mode.hashCode() + i2));
            if (porterDuffColorFilter == null) {
                porterDuffColorFilter = new PorterDuffColorFilter(i, mode);
                Objects.requireNonNull(aVar);
                aVar.put(Integer.valueOf(mode.hashCode() + i2), porterDuffColorFilter);
            }
        }
        return porterDuffColorFilter;
    }

    public final synchronized boolean a(Context context, long j, Drawable drawable) {
        Drawable.ConstantState constantState = drawable.getConstantState();
        if (constantState != null) {
            b.f.e<WeakReference<Drawable.ConstantState>> eVar = this.f892g.get(context);
            if (eVar == null) {
                eVar = new b.f.e<>(10);
                this.f892g.put(context, eVar);
            }
            eVar.g(j, new WeakReference<>(constantState));
            return true;
        }
        return false;
    }

    public final Drawable b(Context context, int i) {
        if (this.f893h == null) {
            this.f893h = new TypedValue();
        }
        TypedValue typedValue = this.f893h;
        context.getResources().getValue(i, typedValue, true);
        long j = (typedValue.assetCookie << 32) | typedValue.data;
        Drawable d2 = d(context, j);
        if (d2 != null) {
            return d2;
        }
        c cVar = this.j;
        LayerDrawable layerDrawable = null;
        if (cVar != null) {
            j.a aVar = (j.a) cVar;
            if (i == R.drawable.abc_cab_background_top_material) {
                layerDrawable = new LayerDrawable(new Drawable[]{e(context, R.drawable.abc_cab_background_internal_bg), e(context, 2131165208)});
            }
        }
        if (layerDrawable != null) {
            layerDrawable.setChangingConfigurations(typedValue.changingConfigurations);
            a(context, j, layerDrawable);
        }
        return layerDrawable;
    }

    public final synchronized Drawable d(Context context, long j) {
        b.f.e<WeakReference<Drawable.ConstantState>> eVar = this.f892g.get(context);
        if (eVar == null) {
            return null;
        }
        WeakReference<Drawable.ConstantState> e2 = eVar.e(j, null);
        if (e2 != null) {
            Drawable.ConstantState constantState = e2.get();
            if (constantState != null) {
                return constantState.newDrawable(context.getResources());
            }
            int b2 = b.f.d.b(eVar.f1751d, eVar.f1753f, j);
            if (b2 >= 0) {
                Object[] objArr = eVar.f1752e;
                Object obj = objArr[b2];
                Object obj2 = b.f.e.f1749b;
                if (obj != obj2) {
                    objArr[b2] = obj2;
                    eVar.f1750c = true;
                }
            }
        }
        return null;
    }

    public synchronized Drawable e(Context context, int i) {
        return f(context, i, false);
    }

    public synchronized Drawable f(Context context, int i, boolean z) {
        Drawable i2;
        if (!this.i) {
            boolean z2 = true;
            this.i = true;
            Drawable e2 = e(context, R.drawable.abc_vector_test);
            if (e2 != null) {
                if (!(e2 instanceof b.a0.a.a.c) && !"android.graphics.drawable.VectorDrawable".equals(e2.getClass().getName())) {
                    z2 = false;
                }
            }
            this.i = false;
            throw new IllegalStateException("This app has been built with an incorrect configuration. Please configure your build for VectorDrawableCompat.");
        }
        i2 = i(context, i);
        if (i2 == null) {
            i2 = b(context, i);
        }
        if (i2 == null) {
            Object obj = b.j.c.a.f2074a;
            i2 = context.getDrawable(i);
        }
        if (i2 != null) {
            i2 = j(context, i, z, i2);
        }
        if (i2 != null) {
            int[] iArr = e0.f829a;
        }
        return i2;
    }

    public synchronized ColorStateList h(Context context, int i) {
        ColorStateList e2;
        b.f.i<ColorStateList> iVar;
        WeakHashMap<Context, b.f.i<ColorStateList>> weakHashMap = this.f889d;
        ColorStateList colorStateList = null;
        e2 = (weakHashMap == null || (iVar = weakHashMap.get(context)) == null) ? null : iVar.e(i, null);
        if (e2 == null) {
            c cVar = this.j;
            if (cVar != null) {
                colorStateList = ((j.a) cVar).c(context, i);
            }
            if (colorStateList != null) {
                if (this.f889d == null) {
                    this.f889d = new WeakHashMap<>();
                }
                b.f.i<ColorStateList> iVar2 = this.f889d.get(context);
                if (iVar2 == null) {
                    iVar2 = new b.f.i<>(10);
                    this.f889d.put(context, iVar2);
                }
                iVar2.a(i, colorStateList);
            }
            e2 = colorStateList;
        }
        return e2;
    }

    public final Drawable i(Context context, int i) {
        int next;
        b.f.h<String, b> hVar = this.f890e;
        if (hVar == null || hVar.isEmpty()) {
            return null;
        }
        b.f.i<String> iVar = this.f891f;
        if (iVar != null) {
            String e2 = iVar.e(i, null);
            if ("appcompat_skip_skip".equals(e2) || (e2 != null && this.f890e.getOrDefault(e2, null) == null)) {
                return null;
            }
        } else {
            this.f891f = new b.f.i<>(10);
        }
        if (this.f893h == null) {
            this.f893h = new TypedValue();
        }
        TypedValue typedValue = this.f893h;
        Resources resources = context.getResources();
        resources.getValue(i, typedValue, true);
        long j = (typedValue.assetCookie << 32) | typedValue.data;
        Drawable d2 = d(context, j);
        if (d2 != null) {
            return d2;
        }
        CharSequence charSequence = typedValue.string;
        if (charSequence != null && charSequence.toString().endsWith(".xml")) {
            try {
                XmlResourceParser xml = resources.getXml(i);
                AttributeSet asAttributeSet = Xml.asAttributeSet(xml);
                while (true) {
                    next = xml.next();
                    if (next == 2 || next == 1) {
                        break;
                    }
                }
                if (next == 2) {
                    String name = xml.getName();
                    this.f891f.a(i, name);
                    b bVar = this.f890e.get(name);
                    if (bVar != null) {
                        d2 = bVar.a(context, xml, asAttributeSet, context.getTheme());
                    }
                    if (d2 != null) {
                        d2.setChangingConfigurations(typedValue.changingConfigurations);
                        a(context, j, d2);
                    }
                } else {
                    throw new XmlPullParserException("No start tag found");
                }
            } catch (Exception e3) {
                Log.e("ResourceManagerInternal", "Exception while inflating drawable", e3);
            }
        }
        if (d2 == null) {
            this.f891f.a(i, "appcompat_skip_skip");
        }
        return d2;
    }

    public final Drawable j(Context context, int i, boolean z, Drawable drawable) {
        ColorStateList h2 = h(context, i);
        PorterDuff.Mode mode = null;
        if (h2 != null) {
            if (e0.a(drawable)) {
                drawable = drawable.mutate();
            }
            drawable.setTintList(h2);
            c cVar = this.j;
            if (cVar != null) {
                j.a aVar = (j.a) cVar;
                if (i == R.drawable.abc_switch_thumb_material) {
                    mode = PorterDuff.Mode.MULTIPLY;
                }
            }
            if (mode != null) {
                drawable.setTintMode(mode);
                return drawable;
            }
            return drawable;
        }
        c cVar2 = this.j;
        if (cVar2 != null) {
            j.a aVar2 = (j.a) cVar2;
            boolean z2 = true;
            if (i == R.drawable.abc_seekbar_track_material) {
                LayerDrawable layerDrawable = (LayerDrawable) drawable;
                Drawable findDrawableByLayerId = layerDrawable.findDrawableByLayerId(16908288);
                int c2 = t0.c(context, R.attr.colorControlNormal);
                PorterDuff.Mode mode2 = j.f862a;
                aVar2.d(findDrawableByLayerId, c2, mode2);
                aVar2.d(layerDrawable.findDrawableByLayerId(16908303), t0.c(context, R.attr.colorControlNormal), mode2);
                aVar2.d(layerDrawable.findDrawableByLayerId(16908301), t0.c(context, R.attr.colorControlActivated), mode2);
            } else if (i == R.drawable.abc_ratingbar_material || i == R.drawable.abc_ratingbar_indicator_material || i == R.drawable.abc_ratingbar_small_material) {
                LayerDrawable layerDrawable2 = (LayerDrawable) drawable;
                Drawable findDrawableByLayerId2 = layerDrawable2.findDrawableByLayerId(16908288);
                int b2 = t0.b(context, R.attr.colorControlNormal);
                PorterDuff.Mode mode3 = j.f862a;
                aVar2.d(findDrawableByLayerId2, b2, mode3);
                aVar2.d(layerDrawable2.findDrawableByLayerId(16908303), t0.c(context, R.attr.colorControlActivated), mode3);
                aVar2.d(layerDrawable2.findDrawableByLayerId(16908301), t0.c(context, R.attr.colorControlActivated), mode3);
            } else {
                z2 = false;
            }
            if (z2) {
                return drawable;
            }
        }
        if (k(context, i, drawable) || !z) {
            return drawable;
        }
        return null;
    }

    /* JADX WARN: Removed duplicated region for block: B:22:0x0052  */
    /* JADX WARN: Removed duplicated region for block: B:29:0x006e  */
    /* JADX WARN: Removed duplicated region for block: B:34:? A[RETURN, SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean k(Context context, int i, Drawable drawable) {
        int i2;
        boolean z;
        int i3;
        boolean z2;
        c cVar = this.j;
        if (cVar != null) {
            j.a aVar = (j.a) cVar;
            Objects.requireNonNull(aVar);
            PorterDuff.Mode mode = j.f862a;
            int i4 = 16842801;
            if (aVar.a(aVar.f865a, i)) {
                i4 = R.attr.colorControlNormal;
            } else if (aVar.a(aVar.f867c, i)) {
                i4 = R.attr.colorControlActivated;
            } else if (aVar.a(aVar.f868d, i)) {
                mode = PorterDuff.Mode.MULTIPLY;
            } else if (i == R.drawable.abc_list_divider_mtrl_alpha) {
                i2 = 16842800;
                i3 = Math.round(40.8f);
                z = true;
                if (z) {
                    if (e0.a(drawable)) {
                        drawable = drawable.mutate();
                    }
                    drawable.setColorFilter(j.c(t0.c(context, i2), mode));
                    if (i3 != -1) {
                        drawable.setAlpha(i3);
                    }
                    z2 = true;
                } else {
                    z2 = false;
                }
                if (z2) {
                    return true;
                }
            } else if (i != R.drawable.abc_dialog_material_background) {
                i2 = 0;
                z = false;
                i3 = -1;
                if (z) {
                }
                if (z2) {
                }
            }
            i2 = i4;
            i3 = -1;
            z = true;
            if (z) {
            }
            if (z2) {
            }
        }
        return false;
    }
}