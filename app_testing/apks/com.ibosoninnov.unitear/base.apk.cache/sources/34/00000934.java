package c.d.c;

import android.content.Context;
import android.content.pm.ApplicationInfo;
import com.google.firebase.platforminfo.LibraryVersionComponent;

/* compiled from: lambda */
/* loaded from: classes.dex */
public final /* synthetic */ class e implements LibraryVersionComponent.VersionExtractor {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ e f4384a = new e();

    @Override // com.google.firebase.platforminfo.LibraryVersionComponent.VersionExtractor
    public final String extract(Object obj) {
        ApplicationInfo applicationInfo = ((Context) obj).getApplicationInfo();
        return applicationInfo != null ? String.valueOf(applicationInfo.minSdkVersion) : "";
    }
}