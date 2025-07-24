package com.google.android.datatransport.runtime.scheduling.persistence;

import android.content.Context;
import com.google.android.datatransport.runtime.dagger.internal.Factory;
import com.google.android.datatransport.runtime.dagger.internal.Preconditions;
import d.a.a;

/* loaded from: classes.dex */
public final class EventStoreModule_PackageNameFactory implements Factory<String> {
    private final a<Context> contextProvider;

    public EventStoreModule_PackageNameFactory(a<Context> aVar) {
        this.contextProvider = aVar;
    }

    public static EventStoreModule_PackageNameFactory create(a<Context> aVar) {
        return new EventStoreModule_PackageNameFactory(aVar);
    }

    public static String packageName(Context context) {
        return (String) Preconditions.checkNotNull(EventStoreModule.packageName(context), "Cannot return null from a non-@Nullable @Provides method");
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.android.datatransport.runtime.dagger.internal.Factory, d.a.a
    public String get() {
        return packageName(this.contextProvider.get());
    }
}