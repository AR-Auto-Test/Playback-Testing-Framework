package c.c.a.m.x.e;

import android.content.Context;
import android.content.res.Resources;
import android.graphics.drawable.Drawable;

/* compiled from: DrawableDecoderCompat.java */
/* loaded from: classes.dex */
public final class a {

    /* renamed from: a  reason: collision with root package name */
    public static volatile boolean f4022a = true;

    public static Drawable a(Context context, Context context2, int i, Resources.Theme theme) {
        try {
            if (f4022a) {
                return b.b.d.a.a.a(theme != null ? new b.b.g.c(context2, theme) : context2, i);
            }
        } catch (Resources.NotFoundException unused) {
        } catch (IllegalStateException e2) {
            if (!context.getPackageName().equals(context2.getPackageName())) {
                Object obj = b.j.c.a.f2074a;
                return context2.getDrawable(i);
            }
            throw e2;
        } catch (NoClassDefFoundError unused2) {
            f4022a = false;
        }
        if (theme == null) {
            theme = context2.getTheme();
        }
        return context2.getResources().getDrawable(i, theme);
    }
}