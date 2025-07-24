package c.d.c;

import android.content.Context;
import android.content.pm.ApplicationInfo;
import com.google.firebase.platforminfo.LibraryVersionComponent;

/* compiled from: lambda */
/* loaded from: classes.dex */
public final /* synthetic */ class d implements LibraryVersionComponent.VersionExtractor {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ d f4383a = new d();

    @Override // com.google.firebase.platforminfo.LibraryVersionComponent.VersionExtractor
    public final String extract(Object obj) {
        ApplicationInfo applicationInfo = ((Context) obj).getApplicationInfo();
        return applicationInfo != null ? String.valueOf(applicationInfo.targetSdkVersion) : "";
    }
}