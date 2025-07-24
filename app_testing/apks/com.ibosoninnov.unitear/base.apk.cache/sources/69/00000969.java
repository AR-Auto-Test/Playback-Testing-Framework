package c.d.c.j;

import com.google.firebase.heartbeatinfo.DefaultHeartBeatController;
import java.util.concurrent.ThreadFactory;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class d implements ThreadFactory {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ d f4449a = new d();

    @Override // java.util.concurrent.ThreadFactory
    public final Thread newThread(Runnable runnable) {
        int i = DefaultHeartBeatController.f5647a;
        return new Thread(runnable, "heartbeat-information-executor");
    }
}