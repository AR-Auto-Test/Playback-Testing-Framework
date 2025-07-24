package b.b.h;

import android.content.Context;
import android.content.res.ColorStateList;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffColorFilter;
import android.graphics.drawable.Drawable;
import android.util.Log;
import android.util.TypedValue;
import b.b.h.n0;
import com.ibosoninnov.unitear.R;

/* compiled from: AppCompatDrawableManager.java */
/* loaded from: classes.dex */
public final class j {

    /* renamed from: a  reason: collision with root package name */
    public static final PorterDuff.Mode f862a = PorterDuff.Mode.SRC_IN;

    /* renamed from: b  reason: collision with root package name */
    public static j f863b;

    /* renamed from: c  reason: collision with root package name */
    public n0 f864c;

    /* compiled from: AppCompatDrawableManager.java */
    /* loaded from: classes.dex */
    public class a implements n0.c {

        /* renamed from: a  reason: collision with root package name */
        public final int[] f865a = {R.drawable.abc_textfield_search_default_mtrl_alpha, R.drawable.abc_textfield_default_mtrl_alpha, R.drawable.abc_ab_share_pack_mtrl_alpha};

        /* renamed from: b  reason: collision with root package name */
        public final int[] f866b = {R.drawable.abc_ic_commit_search_api_mtrl_alpha, R.drawable.abc_seekbar_tick_mark_material, R.drawable.abc_ic_menu_share_mtrl_alpha, R.drawable.abc_ic_menu_copy_mtrl_am_alpha, R.drawable.abc_ic_menu_cut_mtrl_alpha, R.drawable.abc_ic_menu_selectall_mtrl_alpha, R.drawable.abc_ic_menu_paste_mtrl_am_alpha};

        /* renamed from: c  reason: collision with root package name */
        public final int[] f867c = {R.drawable.abc_textfield_activated_mtrl_alpha, R.drawable.abc_textfield_search_activated_mtrl_alpha, 2131165208, R.drawable.abc_text_cursor_material, 2131165265, 2131165267, 2131165269, 2131165266, 2131165268, 2131165270};

        /* renamed from: d  reason: collision with root package name */
        public final int[] f868d = {2131165246, R.drawable.abc_cab_background_internal_bg, R.drawable.abc_menu_hardkey_panel_mtrl_mult};

        /* renamed from: e  reason: collision with root package name */
        public final int[] f869e = {R.drawable.abc_tab_indicator_material, R.drawable.abc_textfield_search_material};

        /* renamed from: f  reason: collision with root package name */
        public final int[] f870f = {R.drawable.abc_btn_check_material, R.drawable.abc_btn_radio_material, R.drawable.abc_btn_check_material_anim, R.drawable.abc_btn_radio_material_anim};

        public final boolean a(int[] iArr, int i) {
            for (int i2 : iArr) {
                if (i2 == i) {
                    return true;
                }
            }
            return false;
        }

        public final ColorStateList b(Context context, int i) {
            int c2 = t0.c(context, R.attr.colorControlHighlight);
            return new ColorStateList(new int[][]{t0.f922b, t0.f924d, t0.f923c, t0.f926f}, new int[]{t0.b(context, R.attr.colorButtonNormal), b.j.d.a.a(c2, i), b.j.d.a.a(c2, i), i});
        }

        public ColorStateList c(Context context, int i) {
            if (i == R.drawable.abc_edit_text_material) {
                ThreadLocal<TypedValue> threadLocal = b.b.d.a.a.f630a;
                return context.getColorStateList(R.color.abc_tint_edittext);
            } else if (i == R.drawable.abc_switch_track_mtrl_alpha) {
                ThreadLocal<TypedValue> threadLocal2 = b.b.d.a.a.f630a;
                return context.getColorStateList(R.color.abc_tint_switch_track);
            } else if (i == R.drawable.abc_switch_thumb_material) {
                int[][] iArr = new int[3];
                int[] iArr2 = new int[3];
                ColorStateList d2 = t0.d(context, R.attr.colorSwitchThumbNormal);
                if (d2 != null && d2.isStateful()) {
                    iArr[0] = t0.f922b;
                    iArr2[0] = d2.getColorForState(iArr[0], 0);
                    iArr[1] = t0.f925e;
                    iArr2[1] = t0.c(context, R.attr.colorControlActivated);
                    iArr[2] = t0.f926f;
                    iArr2[2] = d2.getDefaultColor();
                } else {
                    iArr[0] = t0.f922b;
                    iArr2[0] = t0.b(context, R.attr.colorSwitchThumbNormal);
                    iArr[1] = t0.f925e;
                    iArr2[1] = t0.c(context, R.attr.colorControlActivated);
                    iArr[2] = t0.f926f;
                    iArr2[2] = t0.c(context, R.attr.colorSwitchThumbNormal);
                }
                return new ColorStateList(iArr, iArr2);
            } else if (i == R.drawable.abc_btn_default_mtrl_shape) {
                return b(context, t0.c(context, R.attr.colorButtonNormal));
            } else {
                if (i == R.drawable.abc_btn_borderless_material) {
                    return b(context, 0);
                }
                if (i == R.drawable.abc_btn_colored_material) {
                    return b(context, t0.c(context, R.attr.colorAccent));
                }
                if (i != 2131165258 && i != R.drawable.abc_spinner_textfield_background_material) {
                    if (a(this.f866b, i)) {
                        return t0.d(context, R.attr.colorControlNormal);
                    }
                    if (a(this.f869e, i)) {
                        ThreadLocal<TypedValue> threadLocal3 = b.b.d.a.a.f630a;
                        return context.getColorStateList(R.color.abc_tint_default);
                    } else if (a(this.f870f, i)) {
                        ThreadLocal<TypedValue> threadLocal4 = b.b.d.a.a.f630a;
                        return context.getColorStateList(R.color.abc_tint_btn_checkable);
                    } else if (i == R.drawable.abc_seekbar_thumb_material) {
                        ThreadLocal<TypedValue> threadLocal5 = b.b.d.a.a.f630a;
                        return context.getColorStateList(R.color.abc_tint_seek_thumb);
                    } else {
                        return null;
                    }
                }
                ThreadLocal<TypedValue> threadLocal6 = b.b.d.a.a.f630a;
                return context.getColorStateList(R.color.abc_tint_spinner);
            }
        }

        public final void d(Drawable drawable, int i, PorterDuff.Mode mode) {
            if (e0.a(drawable)) {
                drawable = drawable.mutate();
            }
            if (mode == null) {
                mode = j.f862a;
            }
            drawable.setColorFilter(j.c(i, mode));
        }
    }

    public static synchronized j a() {
        j jVar;
        synchronized (j.class) {
            if (f863b == null) {
                e();
            }
            jVar = f863b;
        }
        return jVar;
    }

    public static synchronized PorterDuffColorFilter c(int i, PorterDuff.Mode mode) {
        PorterDuffColorFilter g2;
        synchronized (j.class) {
            g2 = n0.g(i, mode);
        }
        return g2;
    }

    public static synchronized void e() {
        synchronized (j.class) {
            if (f863b == null) {
                j jVar = new j();
                f863b = jVar;
                jVar.f864c = n0.c();
                n0 n0Var = f863b.f864c;
                a aVar = new a();
                synchronized (n0Var) {
                    n0Var.j = aVar;
                }
            }
        }
    }

    public static void f(Drawable drawable, w0 w0Var, int[] iArr) {
        PorterDuff.Mode mode = n0.f886a;
        if (e0.a(drawable) && drawable.mutate() != drawable) {
            Log.d("ResourceManagerInternal", "Mutated drawable is not the same instance as the input.");
            return;
        }
        boolean z = w0Var.f956d;
        if (!z && !w0Var.f955c) {
            drawable.clearColorFilter();
            return;
        }
        PorterDuffColorFilter porterDuffColorFilter = null;
        ColorStateList colorStateList = z ? w0Var.f953a : null;
        PorterDuff.Mode mode2 = w0Var.f955c ? w0Var.f954b : n0.f886a;
        if (colorStateList != null && mode2 != null) {
            porterDuffColorFilter = n0.g(colorStateList.getColorForState(iArr, 0), mode2);
        }
        drawable.setColorFilter(porterDuffColorFilter);
    }

    public synchronized Drawable b(Context context, int i) {
        return this.f864c.e(context, i);
    }

    public synchronized ColorStateList d(Context context, int i) {
        return this.f864c.h(context, i);
    }
}