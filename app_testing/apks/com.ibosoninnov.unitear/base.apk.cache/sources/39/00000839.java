package c.c.a.m.x.e;

import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.Resources;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import c.c.a.m.p;
import c.c.a.m.r;
import c.c.a.m.v.w;
import com.google.firebase.crashlytics.internal.settings.DefaultSettingsSpiCall;
import java.util.List;

/* compiled from: ResourceDrawableDecoder.java */
/* loaded from: classes.dex */
public class d implements r<Uri, Drawable> {

    /* renamed from: a  reason: collision with root package name */
    public final Context f4024a;

    public d(Context context) {
        this.f4024a = context.getApplicationContext();
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, c.c.a.m.p] */
    @Override // c.c.a.m.r
    public boolean a(Uri uri, p pVar) {
        return uri.getScheme().equals("android.resource");
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    /* JADX DEBUG: Return type fixed from 'c.c.a.m.v.w' to match base method */
    @Override // c.c.a.m.r
    public /* bridge */ /* synthetic */ w<Drawable> b(Uri uri, int i, int i2, p pVar) {
        return c(uri);
    }

    /* JADX DEBUG: Incorrect args count in method signature: (Landroid/net/Uri;IILc/c/a/m/p;)Lc/c/a/m/v/w<Landroid/graphics/drawable/Drawable;>; */
    public w c(Uri uri) {
        Context context;
        int parseInt;
        String authority = uri.getAuthority();
        if (authority.equals(this.f4024a.getPackageName())) {
            context = this.f4024a;
        } else {
            try {
                context = this.f4024a.createPackageContext(authority, 0);
            } catch (PackageManager.NameNotFoundException e2) {
                if (authority.contains(this.f4024a.getPackageName())) {
                    context = this.f4024a;
                } else {
                    throw new IllegalArgumentException(c.b.a.a.a.n("Failed to obtain context or unrecognized Uri format for: ", uri), e2);
                }
            }
        }
        List<String> pathSegments = uri.getPathSegments();
        if (pathSegments.size() == 2) {
            List<String> pathSegments2 = uri.getPathSegments();
            String authority2 = uri.getAuthority();
            String str = pathSegments2.get(0);
            String str2 = pathSegments2.get(1);
            parseInt = context.getResources().getIdentifier(str2, str, authority2);
            if (parseInt == 0) {
                parseInt = Resources.getSystem().getIdentifier(str2, str, DefaultSettingsSpiCall.ANDROID_CLIENT_TYPE);
            }
            if (parseInt == 0) {
                throw new IllegalArgumentException(c.b.a.a.a.n("Failed to find resource id for: ", uri));
            }
        } else if (pathSegments.size() == 1) {
            try {
                parseInt = Integer.parseInt(uri.getPathSegments().get(0));
            } catch (NumberFormatException e3) {
                throw new IllegalArgumentException(c.b.a.a.a.n("Unrecognized Uri format: ", uri), e3);
            }
        } else {
            throw new IllegalArgumentException(c.b.a.a.a.n("Unrecognized Uri format: ", uri));
        }
        Drawable a2 = a.a(this.f4024a, context, parseInt, null);
        if (a2 != null) {
            return new c(a2);
        }
        return null;
    }
}