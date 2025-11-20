#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

using namespace llvm;

namespace {

struct AsmTemplate {
  std::string Asm;
  std::string Constraints;
  Type *ResultType = nullptr;
  std::vector<Type *> ArgTypes;
};

std::optional<AsmTemplate> getTemplate(StringRef Name, LLVMContext &Ctx) {
  auto *I32 = Type::getInt32Ty(Ctx);
  auto *I64 = Type::getInt64Ty(Ctx);

  const static std::unordered_map<std::string, AsmTemplate> Table = {
      {"llvm.pimbp.from_cpu.i32",
       {"movl $1, %eax\n\t"
        ".byte 0x0F, 0x38, 0x90, 0xC0\n\t"
        "movl %eax, $0",
        "=r,r,~{eax}",
        I32,
        {I32}}},
      {"llvm.pimbp.from_cpu.i64",
       {"movq $1, %rax\n\t"
        ".byte 0x48, 0x0F, 0x38, 0x90, 0xC0\n\t"
        "movq %rax, $0",
        "=r,r,~{rax}",
        I64,
        {I64}}},
      {"llvm.pimbp.to_cpu.i32",
       {"movl $1, %eax\n\t"
        ".byte 0x0F, 0x38, 0x91, 0xC0\n\t"
        "movl %eax, $0",
        "=r,r,~{eax}",
        I32,
        {I32}}},
      {"llvm.pimbp.to_cpu.i64",
       {"movq $1, %rax\n\t"
        ".byte 0x48, 0x0F, 0x38, 0x91, 0xC0\n\t"
        "movq %rax, $0",
        "=r,r,~{rax}",
        I64,
        {I64}}},
      {"llvm.pimbp.extsi.i32.i64",
       {"movl $1, %eax\n\t"
        ".byte 0x0F, 0x38, 0x92, 0xC0\n\t"
        "movq %rax, $0",
        "=r,r,~{rax}",
        I64,
        {I32}}},
      {"llvm.pimbp.trunci.i64.i32",
       {"movq $1, %rax\n\t"
        ".byte 0x48, 0x0F, 0x38, 0x93, 0xC0\n\t"
        "movl %eax, $0",
        "=r,r,~{rax}",
        I32,
        {I64}}},
      {"llvm.pimbp.addi.i32",
       {"movl $1, %eax\n\t"
        "movl $2, %ebx\n\t"
        ".byte 0x0F, 0x38, 0xA2, 0xC3\n\t"
        "movl %eax, $0",
        "=r,r,r,~{eax},~{ebx}",
        I32,
        {I32, I32}}},
      {"llvm.pimbp.addi.i64",
       {"movq $1, %rax\n\t"
        "movq $2, %rbx\n\t"
        ".byte 0x48, 0x0F, 0x38, 0xA2, 0xC3\n\t"
        "movq %rax, $0",
        "=r,r,r,~{rax},~{rbx}",
        I64,
        {I64, I64}}},
      {"llvm.pimbp.subi.i32",
       {"movl $1, %eax\n\t"
        "movl $2, %ebx\n\t"
        ".byte 0x0F, 0x38, 0xA3, 0xC3\n\t"
        "movl %eax, $0",
        "=r,r,r,~{eax},~{ebx}",
        I32,
        {I32, I32}}},
      {"llvm.pimbp.subi.i64",
       {"movq $1, %rax\n\t"
        "movq $2, %rbx\n\t"
        ".byte 0x48, 0x0F, 0x38, 0xA3, 0xC3\n\t"
        "movq %rax, $0",
        "=r,r,r,~{rax},~{rbx}",
        I64,
        {I64, I64}}},
      {"llvm.pimbp.andi.i32",
       {"movl $1, %eax\n\t"
        "movl $2, %ebx\n\t"
        ".byte 0x0F, 0x38, 0xA6, 0xC3\n\t"
        "movl %eax, $0",
        "=r,r,r,~{eax},~{ebx}",
        I32,
        {I32, I32}}},
      {"llvm.pimbp.andi.i64",
       {"movq $1, %rax\n\t"
        "movq $2, %rbx\n\t"
        ".byte 0x48, 0x0F, 0x38, 0xA6, 0xC3\n\t"
        "movq %rax, $0",
        "=r,r,r,~{rax},~{rbx}",
        I64,
        {I64, I64}}},
      {"llvm.pimbp.xori.i32",
       {"movl $1, %eax\n\t"
        "movl $2, %ebx\n\t"
        ".byte 0x0F, 0x38, 0xA7, 0xC3\n\t"
        "movl %eax, $0",
        "=r,r,r,~{eax},~{ebx}",
        I32,
        {I32, I32}}},
      {"llvm.pimbp.xori.i64",
       {"movq $1, %rax\n\t"
        "movq $2, %rbx\n\t"
        ".byte 0x48, 0x0F, 0x38, 0xA7, 0xC3\n\t"
        "movq %rax, $0",
        "=r,r,r,~{rax},~{rbx}",
        I64,
        {I64, I64}}},
      {"llvm.pimbp.ori.i32",
       {"movl $1, %eax\n\t"
        "movl $2, %ebx\n\t"
        ".byte 0x0F, 0x38, 0xA8, 0xC3\n\t"
        "movl %eax, $0",
        "=r,r,r,~{eax},~{ebx}",
        I32,
        {I32, I32}}},
      {"llvm.pimbp.ori.i64",
       {"movq $1, %rax\n\t"
        "movq $2, %rbx\n\t"
        ".byte 0x48, 0x0F, 0x38, 0xA8, 0xC3\n\t"
        "movq %rax, $0",
        "=r,r,r,~{rax},~{rbx}",
        I64,
        {I64, I64}}},
      {"llvm.pimbp.shrui.i32",
      {"movl $1, %eax\n\t"
        ".byte 0x0F, 0x38, 0xA4, 0xC1\n\t"
        "movl %eax, $0",
       "=r,r,{cl},~{eax}",
       I32,
       {I32, I32}}},
      {"llvm.pimbp.shrui.i64",
       {"movq $1, %rax\n\t"
        ".byte 0x48, 0x0F, 0x38, 0xA4, 0xC1\n\t"
        "movq %rax, $0",
       "=r,r,{cl},~{rax}",
       I64,
       {I64, I64}}},
      {"llvm.pimbp.shrsi.i32",
       {"movl $1, %eax\n\t"
        ".byte 0x0F, 0x38, 0xA5, 0xC1\n\t"
        "movl %eax, $0",
       "=r,r,{cl},~{eax}",
       I32,
       {I32, I32}}},
      {"llvm.pimbp.shrsi.i64",
       {"movq $1, %rax\n\t"
        ".byte 0x48, 0x0F, 0x38, 0xA5, 0xC1\n\t"
        "movq %rax, $0",
       "=r,r,{cl},~{rax}",
       I64,
       {I64, I64}}},
      {"llvm.pimbp.muli.i64",
       {"movq $1, %rax\n\t"
        "movq $2, %rdx\n\t"
        ".byte 0x48, 0x0F, 0x38, 0xA0, 0xC2\n\t"
        "movq %rax, $0",
        "=r,r,r,~{rax},~{rdx}",
        I64,
        {I64, I64}}},
      {"llvm.pimbp.remsi.i64",
       {"movq $1, %rax\n\t"
        "movq $2, %rdx\n\t"
        ".byte 0x48, 0x0F, 0x38, 0xA1, 0xC2\n\t"
        "movq %rax, $0",
        "=r,r,r,~{rax},~{rdx}",
        I64,
        {I64, I64}}},
      {"llvm.pimbs.from_cpu.i32",
       {"movl $1, %eax\n\t"
        ".byte 0x0F, 0x38, 0xB0, 0xC0\n\t"
        "movl %eax, $0",
        "=r,r,~{eax}",
        I32,
        {I32}}},
      {"llvm.pimbs.from_cpu.i64",
       {"movq $1, %rax\n\t"
        ".byte 0x48, 0x0F, 0x38, 0xB0, 0xC0\n\t"
        "movq %rax, $0",
        "=r,r,~{rax}",
        I64,
        {I64}}},
      {"llvm.pimbs.to_cpu.i32",
       {"movl $1, %eax\n\t"
        ".byte 0x0F, 0x38, 0xB1, 0xC0\n\t"
        "movl %eax, $0",
        "=r,r,~{eax}",
        I32,
        {I32}}},
      {"llvm.pimbs.to_cpu.i64",
       {"movq $1, %rax\n\t"
        ".byte 0x48, 0x0F, 0x38, 0xB1, 0xC0\n\t"
        "movq %rax, $0",
        "=r,r,~{rax}",
        I64,
        {I64}}},
      {"llvm.pimbs.extsi.i32.i64",
       {"movl $1, %eax\n\t"
        ".byte 0x0F, 0x38, 0xB2, 0xC0\n\t"
        "movq %rax, $0",
        "=r,r,~{rax}",
        I64,
        {I32}}},
      {"llvm.pimbs.trunci.i64.i32",
       {"movq $1, %rax\n\t"
        ".byte 0x48, 0x0F, 0x38, 0xB3, 0xC0\n\t"
        "movl %eax, $0",
        "=r,r,~{rax}",
        I32,
        {I64}}},
      {"llvm.pimbs.addi.i32",
       {"movl $1, %eax\n\t"
        "movl $2, %ebx\n\t"
        ".byte 0x0F, 0x38, 0xC2, 0xC3\n\t"
        "movl %eax, $0",
        "=r,r,r,~{eax},~{ebx}",
        I32,
        {I32, I32}}},
      {"llvm.pimbs.addi.i64",
       {"movq $1, %rax\n\t"
        "movq $2, %rbx\n\t"
        ".byte 0x48, 0x0F, 0x38, 0xC2, 0xC3\n\t"
        "movq %rax, $0",
        "=r,r,r,~{rax},~{rbx}",
        I64,
        {I64, I64}}},
      {"llvm.pimbs.subi.i32",
       {"movl $1, %eax\n\t"
        "movl $2, %ebx\n\t"
        ".byte 0x0F, 0x38, 0xC3, 0xC3\n\t"
        "movl %eax, $0",
        "=r,r,r,~{eax},~{ebx}",
        I32,
        {I32, I32}}},
      {"llvm.pimbs.subi.i64",
       {"movq $1, %rax\n\t"
        "movq $2, %rbx\n\t"
        ".byte 0x48, 0x0F, 0x38, 0xC3, 0xC3\n\t"
        "movq %rax, $0",
        "=r,r,r,~{rax},~{rbx}",
        I64,
        {I64, I64}}},
      {"llvm.pimbs.andi.i32",
       {"movl $1, %eax\n\t"
        "movl $2, %ebx\n\t"
        ".byte 0x0F, 0x38, 0xC6, 0xC3\n\t"
        "movl %eax, $0",
        "=r,r,r,~{eax},~{ebx}",
        I32,
        {I32, I32}}},
      {"llvm.pimbs.andi.i64",
       {"movq $1, %rax\n\t"
        "movq $2, %rbx\n\t"
        ".byte 0x48, 0x0F, 0x38, 0xC6, 0xC3\n\t"
        "movq %rax, $0",
        "=r,r,r,~{rax},~{rbx}",
        I64,
        {I64, I64}}},
      {"llvm.pimbs.xori.i32",
       {"movl $1, %eax\n\t"
        "movl $2, %ebx\n\t"
        ".byte 0x0F, 0x38, 0xC7, 0xC3\n\t"
        "movl %eax, $0",
        "=r,r,r,~{eax},~{ebx}",
        I32,
        {I32, I32}}},
      {"llvm.pimbs.xori.i64",
       {"movq $1, %rax\n\t"
        "movq $2, %rbx\n\t"
        ".byte 0x48, 0x0F, 0x38, 0xC7, 0xC3\n\t"
        "movq %rax, $0",
        "=r,r,r,~{rax},~{rbx}",
        I64,
        {I64, I64}}},
      {"llvm.pimbs.ori.i32",
       {"movl $1, %eax\n\t"
        "movl $2, %ebx\n\t"
        ".byte 0x0F, 0x38, 0xC8, 0xC3\n\t"
        "movl %eax, $0",
        "=r,r,r,~{eax},~{ebx}",
        I32,
        {I32, I32}}},
      {"llvm.pimbs.ori.i64",
       {"movq $1, %rax\n\t"
        "movq $2, %rbx\n\t"
        ".byte 0x48, 0x0F, 0x38, 0xC8, 0xC3\n\t"
        "movq %rax, $0",
        "=r,r,r,~{rax},~{rbx}",
        I64,
        {I64, I64}}},
      {"llvm.pimbs.shrui.i32",
      {"movl $1, %eax\n\t"
        ".byte 0x0F, 0x38, 0xC4, 0xC1\n\t"
        "movl %eax, $0",
       "=r,r,{cl},~{eax}",
       I32,
       {I32, I32}}},
      {"llvm.pimbs.shrui.i64",
       {"movq $1, %rax\n\t"
        ".byte 0x48, 0x0F, 0x38, 0xC4, 0xC1\n\t"
        "movq %rax, $0",
       "=r,r,{cl},~{rax}",
       I64,
       {I64, I64}}},
      {"llvm.pimbs.shrsi.i32",
       {"movl $1, %eax\n\t"
        ".byte 0x0F, 0x38, 0xC5, 0xC1\n\t"
        "movl %eax, $0",
       "=r,r,{cl},~{eax}",
       I32,
       {I32, I32}}},
      {"llvm.pimbs.shrsi.i64",
       {"movq $1, %rax\n\t"
        ".byte 0x48, 0x0F, 0x38, 0xC5, 0xC1\n\t"
        "movq %rax, $0",
       "=r,r,{cl},~{rax}",
       I64,
       {I64, I64}}},
      {"llvm.pimbs.muli.i64",
       {"movq $1, %rax\n\t"
        "movq $2, %rdx\n\t"
        ".byte 0x48, 0x0F, 0x38, 0xC0, 0xC2\n\t"
        "movq %rax, $0",
        "=r,r,r,~{rax},~{rdx}",
        I64,
        {I64, I64}}},
      {"llvm.pimbs.remsi.i64",
       {"movq $1, %rax\n\t"
        "movq $2, %rdx\n\t"
        ".byte 0x48, 0x0F, 0x38, 0xC1, 0xC2\n\t"
        "movq %rax, $0",
        "=r,r,r,~{rax},~{rdx}",
        I64,
        {I64, I64}}},
  };

  auto it = Table.find(Name.str());
  if (it == Table.end())
    return std::nullopt;
  return it->second;
}

class LowerPIMIntrinsicsPass : public PassInfoMixin<LowerPIMIntrinsicsPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    auto isToCpuName = [](StringRef Name) {
      return Name == "llvm.pimbp.to_cpu.i32" || Name == "llvm.pimbp.to_cpu.i64" ||
             Name == "llvm.pimbs.to_cpu.i32" || Name == "llvm.pimbs.to_cpu.i64";
    };

    // First, sanity-check that every to_cpu call is already placed immediately
    // after the instruction producing its operand.  The upstream lowering
    // pipeline is expected to enforce this ordering (e.g., via egg_to_mlir).
    for (auto &BB : F) {
      for (auto &I : BB) {
        auto *Call = dyn_cast<CallInst>(&I);
        if (!Call)
          continue;
        auto *Callee = Call->getCalledFunction();
        if (!Callee || !isToCpuName(Callee->getName()))
          continue;
        Value *Operand = Call->getArgOperand(0);
        auto *Producer = dyn_cast<Instruction>(Operand);
        if (!Producer)
          continue;
        if (Producer->getParent() != Call->getParent())
          continue;
        if (Producer->getNextNode() != Call)
          F.getContext().emitError(
              Call, "pim to_cpu must immediately follow the instruction "
                    "producing its operand");
      }
    }

    LLVMContext &Ctx = F.getContext();
    std::vector<CallInst *> ToErase;
    bool Changed = false;

    for (auto &BB : F) {
      for (auto &I : BB) {
        auto *Call = dyn_cast<CallInst>(&I);
        if (!Call)
          continue;

        auto *Callee = Call->getCalledFunction();
        if (!Callee)
          continue;

        auto Template = getTemplate(Callee->getName(), Ctx);
        if (!Template)
          continue;

        const auto &[AsmStr, Constraints, RetTy, ArgTys] = *Template;

        if (Call->getType() != RetTy)
          continue;
        if (Call->arg_size() != ArgTys.size())
          continue;

        bool TypesMatch = true;
        for (size_t idx = 0; idx < ArgTys.size(); ++idx) {
          if (Call->getArgOperand(idx)->getType() != ArgTys[idx]) {
            TypesMatch = false;
            break;
          }
        }
        if (!TypesMatch)
          continue;

        auto *FnTy = FunctionType::get(RetTy, ArgTys, false);
        auto *Inline = InlineAsm::get(FnTy, AsmStr, Constraints,
                                      /*hasSideEffects=*/true,
                                      /*isAlignStack=*/false);
        std::vector<Value *> Args;
        Args.reserve(Call->arg_size());
        for (auto &Op : Call->args())
          Args.push_back(Op);
        auto *NewCall = CallInst::Create(Inline, Args, "", Call);
        NewCall->setCallingConv(Call->getCallingConv());
        Call->replaceAllUsesWith(NewCall);
        ToErase.push_back(Call);
        Changed = true;
      }
    }

    for (auto *CI : ToErase)
      CI->eraseFromParent();

    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

} // namespace

llvm::PassPluginLibraryInfo getLowerPIMIntrinsicsPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "lower-pim-intrinsics", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "lower-pim-intrinsics") {
                    FPM.addPass(LowerPIMIntrinsicsPass());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getLowerPIMIntrinsicsPluginInfo();
}
