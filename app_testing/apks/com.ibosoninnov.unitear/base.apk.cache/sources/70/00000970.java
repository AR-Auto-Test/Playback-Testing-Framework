package c.d.c.k;

import com.google.firebase.FirebaseApp;
import com.google.firebase.components.ComponentContainer;
import com.google.firebase.components.ComponentFactory;
import com.google.firebase.heartbeatinfo.HeartBeatController;
import com.google.firebase.installations.FirebaseInstallations;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class f implements ComponentFactory {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ f f4458a = new f();

    @Override // com.google.firebase.components.ComponentFactory
    public final Object create(ComponentContainer componentContainer) {
        return new FirebaseInstallations((FirebaseApp) componentContainer.get(FirebaseApp.class), componentContainer.getProvider(HeartBeatController.class));
    }
}